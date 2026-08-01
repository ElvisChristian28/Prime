from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Device(db.Model):
    __tablename__ = 'devices'
    
    id = db.Column(db.Integer, primary_key=True)
    device_ip = db.Column(db.String(50), unique=True, nullable=False)
    status = db.Column(db.String(20), nullable=False)
    cpu_usage = db.Column(db.Float)
    temperature = db.Column(db.Float)
    
    # UPGRADE 1: Use db.func.now() so the SQL engine handles the timestamps
    last_updated = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())

    def to_dict(self):
        """Helper to convert the model to a JSON-friendly dictionary."""
        
        # UPGRADE 2: Safety check to prevent crashes if a timestamp is NULL
        if self.last_updated:
            formatted_time = self.last_updated.strftime("%Y-%m-%d %H:%M:%S")
        else:
            formatted_time = "N/A"
            
        return {
            "device_ip": self.device_ip,
            "status": self.status,
            "cpu_usage": self.cpu_usage,
            "temperature": self.temperature,
            "last_updated": formatted_time
        }