from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import folium
import os
import requests
from geopy.geocoders import Nominatim
from database import db, DeliveryRecord
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
import requests
load_dotenv()
tracking_data = {}
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///logistics.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

fuel_model = pickle.load(open('models/fuel_model.pkl', 'rb'))
emission_model = pickle.load(open('models/emission_model.pkl', 'rb'))
lr_model = pickle.load(open('models/lr_model.pkl', 'rb'))
le_vehicle = pickle.load(open('models/le_vehicle.pkl', 'rb'))
le_road = pickle.load(open('models/le_road.pkl', 'rb'))

geolocator = Nominatim(user_agent="carbon_logistics")

routes_config = [
    {'name': 'Highway Route', 'road_type': 'highway', 'traffic_level': 0.8, 'distance_multiplier': 1.1},
    {'name': 'City Route', 'road_type': 'city', 'traffic_level': 1.5, 'distance_multiplier': 0.9},
    {'name': 'Mixed Route', 'road_type': 'mixed', 'traffic_level': 1.0, 'distance_multiplier': 1.0},
]

def geocode_location(location_name):
    try:
        location = geolocator.geocode(location_name)
        if location:
            return (location.latitude, location.longitude)
        return None
    except:
        return None

def get_route_coords(start_coords, end_coords, route_type):
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=full&geometries=geojson"
        response = requests.get(url, timeout=10)
        data = response.json()
        if data['code'] == 'Ok':
            coords = data['routes'][0]['geometry']['coordinates']
            distance = data['routes'][0]['distance'] / 1000
            coords = [(c[1], c[0]) for c in coords]
            return coords, round(distance, 2)
    except:
        pass

    lat_diff = end_coords[0] - start_coords[0]
    lng_diff = end_coords[1] - start_coords[1]

    if route_type == 'highway':
        mid = (start_coords[0] + lat_diff * 0.5 + 0.05, start_coords[1] + lng_diff * 0.5 + 0.05)
    elif route_type == 'city':
        mid = (start_coords[0] + lat_diff * 0.5 - 0.05, start_coords[1] + lng_diff * 0.5 - 0.05)
    else:
        mid = (start_coords[0] + lat_diff * 0.5, start_coords[1] + lng_diff * 0.5)

    coords = [start_coords, mid, end_coords]
    import math
    distance = math.sqrt((lat_diff * 111) ** 2 + (lng_diff * 111) ** 2)
    return coords, round(distance, 2)

def get_ev_stations(lat, lng, radius=50000):
    try:
        url = f"https://api.openchargemap.io/v3/poi/?output=json&latitude={lat}&longitude={lng}&distance=50&distanceunit=km&maxresults=10&compact=true&verbose=false"
        response = requests.get(url, timeout=10)
        stations = response.json()
        result = []
        for station in stations:
            try:
                addr = station.get('AddressInfo', {})
                result.append({
                    'name': addr.get('Title', 'EV Station'),
                    'lat': addr.get('Latitude'),
                    'lng': addr.get('Longitude'),
                    'address': addr.get('AddressLine1', 'N/A')
                })
            except:
                continue
        return result
    except:
        return []

def get_traffic_color(traffic_level):
    if traffic_level <= 0.8:
        return 'green'
    elif traffic_level <= 1.2:
        return 'orange'
    else:
        return 'red'

@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    vehicle = data['vehicle_type']
    cargo_weight = float(data['cargo_weight_kg'])
    source = data.get('source', '')
    destination = data.get('destination', '')

    source_coords = geocode_location(source) if source else (18.5204, 73.8567)
    dest_coords = geocode_location(destination) if destination else (19.0760, 72.8777)

    if not source_coords:
        source_coords = (18.5204, 73.8567)
    if not dest_coords:
        dest_coords = (19.0760, 72.8777)

    results = []
    all_route_coords = {}

    for route in routes_config:
        coords, real_distance = get_route_coords(source_coords, dest_coords, route['road_type'])
        actual_distance = real_distance * route['distance_multiplier']

        v_encoded = le_vehicle.transform([vehicle])[0]
        r_encoded = le_road.transform([route['road_type']])[0]
        features = np.array([[v_encoded, actual_distance, cargo_weight, route['traffic_level'], r_encoded]])

        fuel_rf = round(float(fuel_model.predict(features)[0]), 2)
        fuel_lr = round(float(lr_model.predict(features)[0]), 2)
        emission = round(float(emission_model.predict(features)[0]), 4)

        all_route_coords[route['name']] = coords

        results.append({
            'route_name': route['name'],
            'distance_km': round(actual_distance, 2),
            'traffic_level': route['traffic_level'],
            'fuel_consumed_liters': fuel_rf,
            'fuel_lr_prediction': fuel_lr,
            'carbon_emission_kg': emission,
            'road_type': route['road_type']
        })

    results.sort(key=lambda x: x['carbon_emission_kg'])
    results[0]['recommended'] = True

    # Generate map
    center_lat = (source_coords[0] + dest_coords[0]) / 2
    center_lng = (source_coords[1] + dest_coords[1]) / 2
    m = folium.Map(location=[center_lat, center_lng], zoom_start=8, tiles='CartoDB dark_matter')

    # Draw routes with traffic colors
    for i, route in enumerate(results):
        coords = all_route_coords[route['route_name']]
        traffic_color = get_traffic_color(route['traffic_level'])
        is_recommended = route.get('recommended', False)

        folium.PolyLine(
            coords,
            color='#74c69d' if is_recommended else traffic_color,
            weight=6 if is_recommended else 3,
            opacity=0.9 if is_recommended else 0.6,
            tooltip=folium.Tooltip(
                f"<b>{route['route_name']}</b><br>"
                f"Distance: {route['distance_km']} km<br>"
                f"Traffic: {'🟢 Low' if route['traffic_level'] <= 0.8 else '🟡 Medium' if route['traffic_level'] <= 1.2 else '🔴 High'}<br>"
                f"CO2: {route['carbon_emission_kg']} kg<br>"
                f"Fuel: {route['fuel_consumed_liters']} L"
            )
        ).add_to(m)

    # Start & End markers
    folium.Marker(
        source_coords,
        popup=folium.Popup(f'<b>📦 Start</b><br>{source}', max_width=200),
        icon=folium.Icon(color='blue', icon='play', prefix='fa')
    ).add_to(m)

    folium.Marker(
        dest_coords,
        popup=folium.Popup(f'<b>🏁 End</b><br>{destination}', max_width=200),
        icon=folium.Icon(color='red', icon='flag', prefix='fa')
    ).add_to(m)

    # EV Charging Stations
    ev_stations = get_ev_stations(center_lat, center_lng)
    for station in ev_stations:
        if station['lat'] and station['lng']:
            folium.Marker(
                [station['lat'], station['lng']],
                popup=folium.Popup(
                    f"<b>⚡ {station['name']}</b><br>{station['address']}",
                    max_width=200
                ),
                icon=folium.Icon(color='green', icon='bolt', prefix='fa')
            ).add_to(m)

    # Map Legend
    legend_html = '''
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
         background: rgba(15,25,35,0.95); padding: 15px; border-radius: 10px;
         border: 1px solid #2d3f50; color: white; font-size: 12px;">
        <b style="color:#74c69d;">🗺️ Map Legend</b><br><br>
        <span style="color:#74c69d;">━━</span> Recommended Route<br>
        <span style="color:orange;">━━</span> Medium Traffic<br>
        <span style="color:red;">━━</span> High Traffic<br>
        <span style="color:#00ff00;">⚡</span> EV Charging Station<br>
        <span style="color:#4488ff;">▶</span> Start Point<br>
        <span style="color:red;">⚑</span> End Point
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs('static', exist_ok=True)
    m.save('static/map.html')

    recommended = results[0]
    record = DeliveryRecord(
        vehicle_type=vehicle,
        distance_km=recommended['distance_km'],
        cargo_weight_kg=cargo_weight,
        recommended_route=recommended['route_name'],
        fuel_consumed=recommended['fuel_consumed_liters'],
        carbon_emission=recommended['carbon_emission_kg']
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({
    'routes': results,
    'source': source,
    'destination': destination,
    'source_coords': list(source_coords),
    'dest_coords': list(dest_coords),
    'ev_stations_count': len(ev_stations)
})

@app.route('/history')
def history():
    records = DeliveryRecord.query.order_by(DeliveryRecord.created_at.desc()).limit(20).all()
    return jsonify({'history': [r.to_dict() for r in records]})

@app.route('/report')
def report():
    records = DeliveryRecord.query.all()
    if not records:
        return jsonify({'message': 'No data yet'})

    total_deliveries = len(records)
    total_emission = round(sum(r.carbon_emission for r in records), 4)
    total_fuel = round(sum(r.fuel_consumed for r in records), 2)
    avg_emission = round(total_emission / total_deliveries, 4)
    avg_fuel = round(total_fuel / total_deliveries, 2)
    trees_saved = round(total_emission * 0.0167, 2)

    vehicle_stats = {}
    for r in records:
        if r.vehicle_type not in vehicle_stats:
            vehicle_stats[r.vehicle_type] = {'count': 0, 'emission': 0}
        vehicle_stats[r.vehicle_type]['count'] += 1
        vehicle_stats[r.vehicle_type]['emission'] += r.carbon_emission

    from collections import defaultdict
    weekly = defaultdict(float)
    for r in records:
        day = r.created_at.strftime('%Y-%m-%d')
        weekly[day] += r.carbon_emission
    weekly_data = [{'date': k, 'emission': round(v, 4)} for k, v in sorted(weekly.items())]

    return jsonify({
        'total_deliveries': total_deliveries,
        'total_carbon_emission_kg': total_emission,
        'total_fuel_consumed_liters': total_fuel,
        'avg_emission_per_delivery': avg_emission,
        'avg_fuel_per_delivery': avg_fuel,
        'equivalent_trees_saved': trees_saved,
        'vehicle_stats': vehicle_stats,
        'weekly_data': weekly_data
    })


with app.app_context():
    db.create_all()

@app.route('/fleet', methods=['POST'])
def fleet_optimize():
    data = request.json
    deliveries = data['deliveries']
    vehicle = data['vehicle_type']
    cargo_weight = float(data['cargo_weight_kg'])

    results = []
    total_emission = 0
    total_fuel = 0
    fleet_coords = []

    for i, delivery in enumerate(deliveries):
        source = delivery['source']
        destination = delivery['destination']

        source_coords = geocode_location(source) or (18.5204, 73.8567)
        dest_coords = geocode_location(destination) or (19.0760, 72.8777)

        coords, distance = get_route_coords(source_coords, dest_coords, 'highway')

        v_encoded = le_vehicle.transform([vehicle])[0]
        r_encoded = le_road.transform(['highway'])[0]
        features = np.array([[v_encoded, distance, cargo_weight, 0.8, r_encoded]])

        fuel = round(float(fuel_model.predict(features)[0]), 2)
        emission = round(float(emission_model.predict(features)[0]), 4)

        total_emission += emission
        total_fuel += fuel

        fleet_coords.append({
            'source': source,
            'destination': destination,
            'source_coords': source_coords,
            'dest_coords': dest_coords,
            'coords': coords
        })

        results.append({
            'delivery_no': i + 1,
            'source': source,
            'destination': destination,
            'plate': delivery.get('plate', 'N/A'),
            'distance_km': round(distance, 2),
            'fuel_consumed_liters': fuel,
            'carbon_emission_kg': emission,
            'source_coords': list(source_coords),
            'dest_coords': list(dest_coords),
            'route_coords': [[c[0], c[1]] for c in coords]
        })

    # Generate fleet map
    if fleet_coords:
        center_lat = sum(d['source_coords'][0] for d in fleet_coords) / len(fleet_coords)
        center_lng = sum(d['source_coords'][1] for d in fleet_coords) / len(fleet_coords)
        m = folium.Map(location=[center_lat, center_lng], zoom_start=7, tiles='CartoDB dark_matter')

        colors = ['green', 'blue', 'orange', 'purple', 'red', 'darkblue', 'darkgreen']

        for i, d in enumerate(fleet_coords):
            color = colors[i % len(colors)]
            folium.PolyLine(
                d['coords'],
                color=color,
                weight=4,
                opacity=0.8,
                tooltip=f"Delivery {i+1}: {d['source']} → {d['destination']}"
            ).add_to(m)

            folium.Marker(
                d['source_coords'],
                popup=f"📦 D{i+1} Start: {d['source']}",
                icon=folium.Icon(color='blue', icon='play', prefix='fa')
            ).add_to(m)

            folium.Marker(
                d['dest_coords'],
                popup=f"🏁 D{i+1} End: {d['destination']}",
                icon=folium.Icon(color=color, icon='flag', prefix='fa')
            ).add_to(m)

        os.makedirs('static', exist_ok=True)
        m.save('static/fleet_map.html')

    return jsonify({
        'deliveries': results,
        'total_emission': round(total_emission, 4),
        'total_fuel': round(total_fuel, 2),
        'trees_equivalent': round(total_emission * 0.0167, 2)
    })

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/realtime-data')
def realtime_data():
    records = DeliveryRecord.query.order_by(DeliveryRecord.created_at.desc()).limit(10).all()
    
    total_emission = round(sum(r.carbon_emission for r in records), 4)
    total_fuel = round(sum(r.fuel_consumed for r in records), 2)
    total_deliveries = len(records)
    trees_saved = round(total_emission * 0.0167, 2)
    
    if total_emission < 5:
        health = 'Excellent'
        health_color = '#74c69d'
        health_score = 95
    elif total_emission < 15:
        health = 'Good'
        health_color = '#b7e4c7'
        health_score = 75
    elif total_emission < 30:
        health = 'Average'
        health_color = '#f4a261'
        health_score = 50
    else:
        health = 'Poor'
        health_color = '#e63946'
        health_score = 25

    recent = []
    for r in records:
        recent.append({
            'time': r.created_at.strftime('%H:%M:%S'),
            'vehicle': r.vehicle_type,
            'emission': r.carbon_emission,
            'fuel': r.fuel_consumed,
            'route': r.recommended_route
        })

    return jsonify({
        'total_emission': total_emission,
        'total_fuel': total_fuel,
        'total_deliveries': total_deliveries,
        'trees_saved': trees_saved,
        'health': health,
        'health_color': health_color,
        'health_score': health_score,
        'recent': recent
    })

@app.route('/download-report')
def download_report():
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import inch
    from flask import send_file
    import io
    from datetime import datetime

    records = DeliveryRecord.query.all()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('title', parent=styles['Title'], fontSize=20, textColor=colors.HexColor('#1a472a'), spaceAfter=10)
    heading_style = ParagraphStyle('heading', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#2d6a4f'), spaceAfter=8)
    normal_style = ParagraphStyle('normal', parent=styles['Normal'], fontSize=11, spaceAfter=6)

    story.append(Paragraph("🌿 Carbon Logistics Optimizer", title_style))
    story.append(Paragraph("Sustainability Impact Report", heading_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 0.3 * inch))

    if records:
        total_deliveries = len(records)
        total_emission = round(sum(r.carbon_emission for r in records), 4)
        total_fuel = round(sum(r.fuel_consumed for r in records), 2)
        avg_emission = round(total_emission / total_deliveries, 4)
        avg_fuel = round(total_fuel / total_deliveries, 2)
        trees_saved = round(total_emission * 0.0167, 2)

        story.append(Paragraph("📊 Summary Statistics", heading_style))
        summary_data = [
            ['Metric', 'Value'],
            ['Total Deliveries', str(total_deliveries)],
            ['Total CO2 Emissions', f'{total_emission} kg'],
            ['Total Fuel Consumed', f'{total_fuel} liters'],
            ['Avg Emission per Delivery', f'{avg_emission} kg'],
            ['Avg Fuel per Delivery', f'{avg_fuel} liters'],
            ['Equivalent Trees Saved', str(trees_saved)],
        ]

        summary_table = Table(summary_data, colWidths=[3 * inch, 3 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a472a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f7f0'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#2d6a4f')),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3 * inch))

        story.append(Paragraph("📋 Delivery History", heading_style))
        history_data = [['#', 'Vehicle', 'Distance (km)', 'Cargo (kg)', 'Route', 'Fuel (L)', 'CO2 (kg)', 'Grade']]

        for i, r in enumerate(records):
            emission = r.carbon_emission
            if emission < 0.5:
                grade = 'A'
            elif emission < 1.5:
                grade = 'B'
            else:
                grade = 'C'

            history_data.append([
                str(i + 1),
                r.vehicle_type,
                str(r.distance_km),
                str(r.cargo_weight_kg),
                r.recommended_route,
                str(r.fuel_consumed),
                str(r.carbon_emission),
                grade
            ])

        history_table = Table(history_data, colWidths=[0.4*inch, 0.8*inch, 1*inch, 0.8*inch, 1.2*inch, 0.7*inch, 0.7*inch, 0.5*inch])
        history_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a472a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f7f0'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#2d6a4f')),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(history_table)

    else:
        story.append(Paragraph("No delivery data available yet.", normal_style))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Generated by Carbon Logistics Optimizer — AI-powered eco-friendly route recommendations", ParagraphStyle('footer', parent=styles['Normal'], fontSize=9, textColor=colors.grey)))

    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f'carbon_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf', mimetype='application/pdf')    

@app.route('/autocomplete')
def autocomplete():
    query = request.args.get('q', '')
    if not query or len(query) < 2:
        return jsonify([])
    try:
        url = f"https://nominatim.openstreetmap.org/search"
        headers = {'User-Agent': 'carbon_logistics_app_v2'}
        params = {
            'q': query,
            'format': 'json',
            'limit': 7,
            'addressdetails': 1,
            'countrycodes': 'in'
        }
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        results = []
        for item in data:
            results.append({
                'display_name': item['display_name'],
                'lat': item['lat'],
                'lng': item['lon']
            })
        return jsonify(results)
    except Exception as e:
        return jsonify([])

@app.route('/journey-time')
def journey_time():
    slat = request.args.get('slat')
    slng = request.args.get('slng')
    dlat = request.args.get('dlat')
    dlng = request.args.get('dlng')
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{slng},{slat};{dlng},{dlat}?overview=false"
        response = requests.get(url, timeout=10)
        data = response.json()
        if data['code'] == 'Ok':
            duration_seconds = data['routes'][0]['duration']
            distance_km = round(data['routes'][0]['distance'] / 1000, 2)
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            return jsonify({
                'duration_seconds': duration_seconds,
                'hours': hours,
                'minutes': minutes,
                'distance_km': distance_km,
                'formatted': f"{hours}h {minutes}m"
            })
    except:
        pass
    return jsonify({'error': 'Could not calculate journey time'})

@app.route('/recommend-vehicle', methods=['POST'])
def recommend_vehicle():
    data = request.json
    distance = float(data['distance_km'])
    cargo_weight = float(data['cargo_weight_kg'])
    priority = data['priority']

    vehicles = ['bike', 'car', 'van', 'truck']
    results = []

    for vehicle in vehicles:
        v_encoded = le_vehicle.transform([vehicle])[0]
        r_encoded = le_road.transform(['highway'])[0]
        features = np.array([[v_encoded, distance, cargo_weight, 0.8, r_encoded]])

        fuel = round(float(fuel_model.predict(features)[0]), 2)
        emission = round(float(emission_model.predict(features)[0]), 4)
        grade = 'A' if emission < 0.5 else 'B' if emission < 1.5 else 'C'

        cost_per_liter = 100
        fuel_cost = round(fuel * cost_per_liter, 2)
        time_hours = round(distance / (60 if vehicle == 'bike' else 80 if vehicle == 'car' else 70 if vehicle == 'van' else 60), 2)

        results.append({
            'vehicle': vehicle,
            'fuel_liters': fuel,
            'emission_kg': emission,
            'grade': grade,
            'fuel_cost_inr': fuel_cost,
            'time_hours': time_hours
        })

    if priority == 'eco':
        results.sort(key=lambda x: x['emission_kg'])
    elif priority == 'cost':
        results.sort(key=lambda x: x['fuel_cost_inr'])
    elif priority == 'time':
        results.sort(key=lambda x: x['time_hours'])

    results[0]['recommended'] = True
    return jsonify({'vehicles': results})

@app.route('/carbon-offset', methods=['POST'])
def carbon_offset():
    data = request.json
    emission_kg = float(data['emission_kg'])

    cost_per_kg = 0.015
    total_cost_usd = round(emission_kg * cost_per_kg, 4)
    total_cost_inr = round(total_cost_usd * 83, 2)
    trees_needed = round(emission_kg * 0.0167, 4)
    km_avoided = round(emission_kg / 0.21, 2)
    bulbs_equivalent = round(emission_kg * 1.2, 2)

    return jsonify({
        'emission_kg': emission_kg,
        'offset_cost_usd': total_cost_usd,
        'offset_cost_inr': total_cost_inr,
        'trees_needed': trees_needed,
        'km_car_avoided': km_avoided,
        'led_bulbs_hours': bulbs_equivalent
    })

@app.route('/ev-simulator', methods=['POST'])
def ev_simulator():
    data = request.json
    distance = float(data['distance_km'])
    cargo_weight = float(data['cargo_weight_kg'])
    current_vehicle = data['vehicle_type']

    v_encoded = le_vehicle.transform([current_vehicle])[0]
    r_encoded = le_road.transform(['highway'])[0]
    features = np.array([[v_encoded, distance, cargo_weight, 0.8, r_encoded]])

    current_fuel = round(float(fuel_model.predict(features)[0]), 2)
    current_emission = round(float(emission_model.predict(features)[0]), 4)
    current_cost = round(current_fuel * 100, 2)

    ev_emission = round(distance * 0.05 * (cargo_weight / 1000 + 1), 4)
    ev_energy_kwh = round(distance * 0.2 * (cargo_weight / 1000 + 1), 2)
    ev_cost = round(ev_energy_kwh * 8, 2)

    emission_saved = round(current_emission - ev_emission, 4)
    cost_saved = round(current_cost - ev_cost, 2)
    percent_reduction = round((emission_saved / current_emission) * 100, 1) if current_emission > 0 else 0

    return jsonify({
        'current_vehicle': current_vehicle,
        'current_fuel_liters': current_fuel,
        'current_emission_kg': current_emission,
        'current_cost_inr': current_cost,
        'ev_emission_kg': ev_emission,
        'ev_energy_kwh': ev_energy_kwh,
        'ev_cost_inr': ev_cost,
        'emission_saved_kg': emission_saved,
        'cost_saved_inr': cost_saved,
        'percent_reduction': percent_reduction
    })

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    message = data['message']
    history = data.get('history', [])

    records = DeliveryRecord.query.all()
    total_deliveries = len(records)
    total_emission = round(sum(r.carbon_emission for r in records), 4) if records else 0
    total_fuel = round(sum(r.fuel_consumed for r in records), 2) if records else 0
    avg_emission = round(total_emission / total_deliveries, 4) if total_deliveries > 0 else 0

    system_prompt = f"""You are an expert Carbon Logistics AI Assistant for a smart delivery optimization system. You help logistics companies reduce carbon emissions.

Current system data:
- Total deliveries made: {total_deliveries}
- Total CO2 emissions: {total_emission} kg
- Total fuel consumed: {total_fuel} liters
- Average emission per delivery: {avg_emission} kg

You can answer ANY question the user asks - about logistics, carbon emissions, sustainability, routes, vehicles, fuel, climate change, EV vehicles, carbon offsetting, supply chain, or any general topic.

For logistics specific questions, use the real data above.
Always be helpful, friendly and provide detailed accurate answers.
Use emojis to make responses engaging.
Keep responses concise but informative."""

    messages = [{"role": "system", "content": system_prompt}]

    for h in history[-6:]:
        messages.append({"role": h['role'], "content": h['content']})

    messages.append({"role": "user", "content": message})

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Sorry, I'm having trouble connecting. Error: {str(e)}"

    return jsonify({'reply': reply})

@app.route('/track', methods=['POST'])
def track_vehicle():
    data = request.json
    plate = data['plate']
    lat = data['lat']
    lng = data['lng']

    tracking_data[plate] = {
        'lat': lat,
        'lng': lng,
        'timestamp': datetime.utcnow().strftime('%H:%M:%S')
    }
    return jsonify({'status': 'updated', 'plate': plate})

@app.route('/tracking-data')
def get_tracking_data():
    return jsonify(tracking_data)

@app.route('/tracking')
def tracking():
    return render_template('tracking.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/app')
def main_app():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/dashboard-data')
def dashboard_data():
    records = DeliveryRecord.query.all()
    total_deliveries = len(records)
    total_emission = round(sum(r.carbon_emission for r in records), 4) if records else 0
    total_fuel = round(sum(r.fuel_consumed for r in records), 2) if records else 0
    avg_emission = round(total_emission / total_deliveries, 4) if total_deliveries > 0 else 0

    grade_a = len([r for r in records if r.carbon_emission < 0.5])
    grade_b = len([r for r in records if 0.5 <= r.carbon_emission < 1.5])
    grade_c = len([r for r in records if r.carbon_emission >= 1.5])

    vehicle_stats = {}
    for r in records:
        if r.vehicle_type not in vehicle_stats:
            vehicle_stats[r.vehicle_type] = {'count': 0, 'emission': 0, 'fuel': 0}
        vehicle_stats[r.vehicle_type]['count'] += 1
        vehicle_stats[r.vehicle_type]['emission'] += r.carbon_emission
        vehicle_stats[r.vehicle_type]['fuel'] += r.fuel_consumed

    weekly = {}
    for r in records:
        day = r.created_at.strftime('%Y-%m-%d')
        if day not in weekly:
            weekly[day] = {'emission': 0, 'count': 0}
        weekly[day]['emission'] += r.carbon_emission
        weekly[day]['count'] += 1

    weekly_data = [{'date': k, 'emission': round(v['emission'], 4), 'count': v['count']} for k, v in sorted(weekly.items())[-14:]]

    trees_saved = round(total_emission * 0.0167, 2)
    carbon_score = max(0, min(100, round(100 - (avg_emission * 20), 1))) if avg_emission > 0 else 100

    return jsonify({
        'total_deliveries': total_deliveries,
        'total_emission': total_emission,
        'total_fuel': total_fuel,
        'avg_emission': avg_emission,
        'grade_a': grade_a,
        'grade_b': grade_b,
        'grade_c': grade_c,
        'vehicle_stats': vehicle_stats,
        'weekly_data': weekly_data,
        'trees_saved': trees_saved,
        'carbon_score': carbon_score
    })

@app.route('/predictive')
def predictive():
    return render_template('predictive.html')

@app.route('/predictive-data')
def predictive_data():
    records = DeliveryRecord.query.all()

    if len(records) < 2:
        return jsonify({'error': 'Not enough data. Please make at least 2 deliveries first!'})

    # Historical daily data
    daily = {}
    for r in records:
        day = r.created_at.strftime('%Y-%m-%d')
        if day not in daily:
            daily[day] = {'emission': 0, 'fuel': 0, 'count': 0}
        daily[day]['emission'] += r.carbon_emission
        daily[day]['fuel'] += r.fuel_consumed
        daily[day]['count'] += 1

    sorted_days = sorted(daily.keys())
    emissions = [daily[d]['emission'] for d in sorted_days]
    fuels = [daily[d]['fuel'] for d in sorted_days]
    counts = [daily[d]['count'] for d in sorted_days]

    # Simple moving average prediction for next 7 days
    avg_emission = sum(emissions[-7:]) / min(len(emissions), 7)
    avg_fuel = sum(fuels[-7:]) / min(len(fuels), 7)
    avg_count = sum(counts[-7:]) / min(len(counts), 7)

    # Trend calculation
    if len(emissions) >= 2:
        trend = (emissions[-1] - emissions[0]) / len(emissions)
    else:
        trend = 0

    from datetime import datetime, timedelta
    last_date = datetime.strptime(sorted_days[-1], '%Y-%m-%d')
    future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(7)]
    future_emissions = [round(max(0, avg_emission + trend * i), 4) for i in range(7)]
    future_fuels = [round(max(0, avg_fuel + trend * i * 0.5), 2) for i in range(7)]

    # Vehicle recommendation based on history
    vehicle_emission = {}
    for r in records:
        if r.vehicle_type not in vehicle_emission:
            vehicle_emission[r.vehicle_type] = []
        vehicle_emission[r.vehicle_type].append(r.carbon_emission)

    vehicle_avg = {v: round(sum(e)/len(e), 4) for v, e in vehicle_emission.items()}
    best_vehicle = min(vehicle_avg, key=vehicle_avg.get) if vehicle_avg else 'bike'

    # Monthly forecast
    monthly_emission = round(avg_emission * 30, 2)
    monthly_fuel = round(avg_fuel * 30, 2)
    monthly_cost = round(monthly_fuel * 100, 2)
    trees_needed = round(monthly_emission * 0.0167, 2)

    # Savings potential
    worst_vehicle = max(vehicle_avg, key=vehicle_avg.get) if vehicle_avg else 'truck'
    savings_potential = round((vehicle_avg.get(worst_vehicle, 0) - vehicle_avg.get(best_vehicle, 0)) * avg_count * 30, 4)

    return jsonify({
        'historical_dates': sorted_days,
        'historical_emissions': emissions,
        'historical_fuels': fuels,
        'future_dates': future_dates,
        'future_emissions': future_emissions,
        'future_fuels': future_fuels,
        'best_vehicle': best_vehicle,
        'vehicle_avg': vehicle_avg,
        'monthly_emission': monthly_emission,
        'monthly_fuel': monthly_fuel,
        'monthly_cost': monthly_cost,
        'trees_needed': trees_needed,
        'savings_potential': savings_potential,
        'avg_daily_emission': round(avg_emission, 4),
        'avg_daily_count': round(avg_count, 1)
    })

@app.route('/scorecard')
def scorecard():
    return render_template('scorecard.html')

@app.route('/scorecard-data')
def scorecard_data():
    records = DeliveryRecord.query.all()
    total_deliveries = len(records)
    total_emission = round(sum(r.carbon_emission for r in records), 4) if records else 0
    total_fuel = round(sum(r.fuel_consumed for r in records), 2) if records else 0
    avg_emission = round(total_emission / total_deliveries, 4) if total_deliveries > 0 else 0

    # Carbon score 0-100
    carbon_score = max(0, min(100, round(100 - (avg_emission * 20), 1))) if avg_emission > 0 else 100

    # Industry average comparison
    industry_avg = 1.2
    comparison = round(((industry_avg - avg_emission) / industry_avg) * 100, 1) if avg_emission > 0 else 100
    better_than_industry = avg_emission < industry_avg

    # Monthly data
    monthly = {}
    for r in records:
        month = r.created_at.strftime('%Y-%m')
        if month not in monthly:
            monthly[month] = {'emission': 0, 'count': 0}
        monthly[month]['emission'] += r.carbon_emission
        monthly[month]['count'] += 1

    monthly_data = [{'month': k, 'emission': round(v['emission'], 4), 'count': v['count'], 'score': max(0, min(100, round(100 - (v['emission']/v['count'] * 20), 1)))} for k, v in sorted(monthly.items())]

    # Grade
    if carbon_score >= 80:
        grade = 'A'
        grade_label = 'Excellent'
    elif carbon_score >= 60:
        grade = 'B'
        grade_label = 'Good'
    elif carbon_score >= 40:
        grade = 'C'
        grade_label = 'Average'
    else:
        grade = 'D'
        grade_label = 'Needs Improvement'

    # Badges
    badges = []
    if total_deliveries >= 1:
        badges.append({'icon': '🚀', 'name': 'First Delivery', 'desc': 'Made your first delivery!'})
    if total_deliveries >= 10:
        badges.append({'icon': '📦', 'name': 'Delivery Pro', 'desc': '10+ deliveries completed!'})
    if avg_emission < 0.5:
        badges.append({'icon': '🌿', 'name': 'Eco Warrior', 'desc': 'Average emission below 0.5kg!'})
    if better_than_industry:
        badges.append({'icon': '🏆', 'name': 'Industry Leader', 'desc': 'Better than industry average!'})
    if carbon_score >= 80:
        badges.append({'icon': '⭐', 'name': 'Green Champion', 'desc': 'Carbon score above 80!'})

    return jsonify({
        'carbon_score': carbon_score,
        'grade': grade,
        'grade_label': grade_label,
        'total_deliveries': total_deliveries,
        'total_emission': total_emission,
        'total_fuel': total_fuel,
        'avg_emission': avg_emission,
        'industry_avg': industry_avg,
        'comparison': comparison,
        'better_than_industry': better_than_industry,
        'monthly_data': monthly_data,
        'badges': badges,
        'trees_saved': round(total_emission * 0.0167, 2)
    })

@app.route('/voice-command', methods=['POST'])
def voice_command():
    data = request.json
    command = data['command'].lower()

    response = {
        'action': None,
        'message': '',
        'data': {}
    }

    if any(word in command for word in ['open map', 'show map', 'map kholo']):
        response['action'] = 'navigate'
        response['data']['page'] = 'home'
        response['message'] = 'Opening the map for you!'

    elif any(word in command for word in ['open report', 'show report', 'report dikhao']):
        response['action'] = 'navigate'
        response['data']['page'] = 'report'
        response['message'] = 'Opening sustainability report!'

    elif any(word in command for word in ['open history', 'show history', 'history dikhao']):
        response['action'] = 'navigate'
        response['data']['page'] = 'history'
        response['message'] = 'Opening delivery history!'

    elif any(word in command for word in ['open fleet', 'show fleet', 'fleet dikhao']):
        response['action'] = 'navigate'
        response['data']['page'] = 'fleet'
        response['message'] = 'Opening fleet optimizer!'

    elif any(word in command for word in ['open tools', 'show tools', 'tools dikhao']):
        response['action'] = 'navigate'
        response['data']['page'] = 'tools'
        response['message'] = 'Opening tools page!'

    elif any(word in command for word in ['dark mode', 'dark theme', 'dark karo']):
        response['action'] = 'theme'
        response['data']['theme'] = 'dark'
        response['message'] = 'Switching to dark mode!'

    elif any(word in command for word in ['light mode', 'light theme', 'light karo']):
        response['action'] = 'theme'
        response['data']['theme'] = 'light'
        response['message'] = 'Switching to light mode!'

    elif any(word in command for word in ['hindi', 'hindi mein', 'hindi me']):
        response['action'] = 'language'
        response['data']['lang'] = 'hi'
        response['message'] = 'हिंदी में बदल रहे हैं!'

    elif any(word in command for word in ['english', 'english mein', 'angrezi']):
        response['action'] = 'language'
        response['data']['lang'] = 'en'
        response['message'] = 'Switching to English!'

    elif any(word in command for word in ['stats', 'statistics', 'show stats', 'kitni delivery']):
        records = DeliveryRecord.query.all()
        total = len(records)
        emission = round(sum(r.carbon_emission for r in records), 4) if records else 0
        response['action'] = 'speak'
        response['message'] = f'You have made {total} deliveries with total CO2 emissions of {emission} kilograms!'

    elif any(word in command for word in ['help', 'kya kar sakte', 'commands', 'what can you do']):
        response['action'] = 'speak'
        response['message'] = 'You can say: Open map, Open report, Open history, Open fleet, Open tools, Dark mode, Light mode, Hindi, English, or Show stats!'

    elif any(word in command for word in ['hello', 'hi', 'hey', 'namaste', 'hii']):
        response['action'] = 'speak'
        response['message'] = 'Hello! I am your Carbon Logistics Voice Assistant! Say help to know what I can do!'

    elif 'route' in command or 'rasta' in command:
        parts = command.replace('find route from', '').replace('route from', '').split(' to ')
        if len(parts) == 2:
            source = parts[0].strip()
            dest = parts[1].strip()
            response['action'] = 'fill_route'
            response['data']['source'] = source
            response['data']['destination'] = dest
            response['message'] = f'Finding route from {source} to {dest}!'
        else:
            response['action'] = 'speak'
            response['message'] = 'Please say: Find route from City A to City B'

    else:
        response['action'] = 'speak'
        response['message'] = f'I heard: {command}. Try saying help to see available commands!'

    return jsonify(response)

@app.route('/co2map')
def co2map():
    return render_template('co2map.html')

@app.route('/co2-live-data')
def co2_live_data():
    co2_api_key = "your_actual_key_here"
    
    countries = [
        {"name": "India", "code": "IN", "lat": 20.59, "lng": 78.96},
        {"name": "USA", "code": "US", "lat": 37.09, "lng": -95.71},
        {"name": "Germany", "code": "DE", "lat": 51.16, "lng": 10.45},
        {"name": "France", "code": "FR", "lat": 46.22, "lng": 2.21},
        {"name": "UK", "code": "GB", "lat": 55.37, "lng": -3.43},
        {"name": "Australia", "code": "AU", "lat": -25.27, "lng": 133.77},
        {"name": "Japan", "code": "JP", "lat": 36.20, "lng": 138.25},
        {"name": "Brazil", "code": "BR", "lat": -14.23, "lng": -51.92},
        {"name": "Canada", "code": "CA", "lat": 56.13, "lng": -106.34},
        {"name": "South Africa", "code": "ZA", "lat": -30.55, "lng": 22.93},
    ]

    results = []
    for country in countries:
        try:
            url = f"https://api.co2signal.com/v1/latest?countryCode={country['code']}"
            res = requests.get(url, headers={
                'auth-token': co2_api_key
            }, timeout=8)
            print(f"{country['name']} status: {res.status_code}")
            print(f"{country['name']} response: {res.text[:200]}")
            
            if res.status_code == 200:
                data = res.json()
                print(f"{country['name']} data keys: {data.keys()}")
                intensity = data.get('data', {}).get('carbonIntensity', 0)
                fossil = data.get('data', {}).get('fossilFuelPercentage', 0)
                results.append({
                    'name': country['name'],
                    'code': country['code'],
                    'lat': country['lat'],
                    'lng': country['lng'],
                    'carbon_intensity': round(float(intensity), 1) if intensity else 0,
                    'fossil_percent': round(float(fossil), 1) if fossil else 0,
                    'status': 'live'
                })
            else:
                results.append({
                    'name': country['name'],
                    'code': country['code'],
                    'lat': country['lat'],
                    'lng': country['lng'],
                    'carbon_intensity': 0,
                    'fossil_percent': 0,
                    'status': 'unavailable'
                })
        except Exception as e:
            print(f"{country['name']} error: {e}")
            results.append({
                'name': country['name'],
                'code': country['code'],
                'lat': country['lat'],
                'lng': country['lng'],
                'carbon_intensity': 0,
                'fossil_percent': 0,
                'status': 'unavailable'
            })

    records = DeliveryRecord.query.all()
    your_emission = round(sum(r.carbon_emission for r in records), 4) if records else 0
    your_deliveries = len(records)

    return jsonify({
        'countries': results,
        'your_emission': your_emission,
        'your_deliveries': your_deliveries
    })


if __name__ == '__main__':
    app.run(debug=True)
