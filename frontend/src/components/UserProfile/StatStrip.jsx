export default function StatStrip({ features }) {
  const stats = [
    { label: "Avg session", value: features?.avg_session_length?.toFixed(1) ?? "0.0" },
    { label: "Recency bias", value: `${Math.round((features?.recency_bias_score ?? 0) * 100)}%` }
  ];
  return (
    <div className="grid grid-cols-2 gap-3">
      {stats.map((stat) => (
        <div key={stat.label} className="rounded-md border border-slate-200 bg-white p-3">
          <div className="text-xs uppercase tracking-wide text-slate-500">{stat.label}</div>
          <div className="mt-1 text-2xl font-semibold text-slate-950">{stat.value}</div>
        </div>
      ))}
    </div>
  );
}
