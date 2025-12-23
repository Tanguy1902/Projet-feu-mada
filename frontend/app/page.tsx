// src/app/page.tsx
'use client';
import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { fetchHighRiskAlerts } from '@/services/api';

// On charge la carte de manière dynamique pour éviter les erreurs SSR de Leaflet
const Map = dynamic(() => import('@/components/Map'), { ssr: false });

export default function Home() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await fetchHighRiskAlerts();
        setAlerts(data);
      } catch (error) {
        console.error("Erreur lors du chargement des alertes:", error);
      } finally {
        setLoading(false);
      }
    };
    loadData();

    // Rafraîchir toutes les 5 minutes pour le temps réel
    const interval = setInterval(loadData, 300000);
    return () => clearInterval(interval);
  }, []);

  return (
    <main className="flex h-screen w-full bg-slate-900 text-white overflow-hidden">
      {/* Barre latérale (AlertPanel) */}
      <div className="w-1/4 h-full p-6 border-r border-slate-700 overflow-y-auto">
        <h1 className="text-2xl font-bold mb-6 text-red-500">Mada Fire Guard</h1>

        <div className="bg-slate-800 p-4 rounded-lg mb-6">
          <p className="text-sm text-slate-400">Alertes actives (&ge; 0.8)</p>
          <p className="text-4xl font-mono">{alerts.length}</p>
        </div>

        <div className="space-y-4">
          <h2 className="text-lg font-semibold border-b border-slate-700 pb-2">Dernières Détections</h2>
          {loading ? <p>Chargement...</p> : alerts.map((alert: any, i: number) => (
            <div key={i} className="p-3 bg-slate-800 rounded border-l-4 border-red-500">
              <p className="font-bold">Risque: {(alert.risk_score * 100).toFixed(0)}%</p>
              <p className="text-xs text-slate-400">Lat: {alert.latitude} | Lon: {alert.longitude}</p>
              <p className="text-xs text-slate-400">{new Date(alert.timestamp).toLocaleString()}</p>
            </div>
          ))}
          {!loading && alerts.length === 0 && <p className="text-slate-500">Aucun risque critique détecté.</p>}
        </div>
      </div>

      {/* Zone de la Carte */}
      <div className="flex-1 h-full relative">
        <Map alerts={alerts} />
      </div>
    </main>
  );
}