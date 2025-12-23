// src/components/Map.tsx
'use client';
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

export default function Map({ alerts }: { alerts: any[] }) {
    const centerMadagascar: [number, number] = [-18.7669, 46.8691];

    return (
        <MapContainer center={centerMadagascar} zoom={6} style={{ height: '100vh', width: '100%' }}>
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

            {alerts.map((alert, index) => (
                <Circle
                    key={index}
                    center={[alert.latitude, alert.longitude]}
                    radius={5000} // Rayon de 5km
                    pathOptions={{ color: 'red', fillColor: 'red', fillOpacity: 0.5 }}
                >
                    <Popup>
                        <strong>Risque de feu : {(alert.risk_score * 100).toFixed(1)}%</strong><br />
                        Température : {alert.temp_max}°C
                    </Popup>
                </Circle>
            ))}
        </MapContainer>
    );
}