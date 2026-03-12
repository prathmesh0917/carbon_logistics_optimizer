from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class DeliveryRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vehicle_type = db.Column(db.String(20), nullable=False)
    distance_km = db.Column(db.Float, nullable=False)
    cargo_weight_kg = db.Column(db.Float, nullable=False)
    recommended_route = db.Column(db.String(50), nullable=False)
    fuel_consumed = db.Column(db.Float, nullable=False)
    carbon_emission = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'vehicle_type': self.vehicle_type,
            'distance_km': self.distance_km,
            'cargo_weight_kg': self.cargo_weight_kg,
            'recommended_route': self.recommended_route,
            'fuel_consumed': self.fuel_consumed,
            'carbon_emission': self.carbon_emission,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }