// src/services/api.ts
export const fetchHighRiskAlerts = async () => {
    const response = await fetch('http://127.0.0.1:8000/api/v1/alerts/high-risk');
    if (!response.ok) throw new Error('Erreur réseau');
    return response.json();
};