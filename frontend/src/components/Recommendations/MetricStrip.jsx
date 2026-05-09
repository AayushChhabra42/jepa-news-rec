function metricColor(name, value) {
  if (name === "AUC") {
    if (value > 0.67) return "bg-emerald-500";
    if (value > 0.63) return "bg-amber-500";
    return "bg-red-500";
  }
  if (value > 0.5) return "bg-emerald-500";
  if (value > 0.25) return "bg-amber-500";
  return "bg-red-500";
}

export default function MetricStrip({ metrics }) {
  const items = [
    ["AUC", metrics?.auc],
    ["MRR", metrics?.mrr],
    ["nDCG@5", metrics?.ndcg5],
    ["nDCG@10", metrics?.ndcg10]
  ];
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {items.map(([label, value]) => {
        const safeValue = Number.isFinite(value) ? value : 0;
        return (
          <div key={label} className="rounded-md border border-slate-200 bg-white p-3">
            <div className="flex items-center justify-between text-xs text-slate-500">
              <span>{label}</span>
              <span className="font-semibold text-slate-900">{safeValue.toFixed(3)}</span>
            </div>
            <div className="mt-2 h-1.5 rounded bg-slate-100">
              <div className={`h-full rounded ${metricColor(label, safeValue)}`} style={{ width: `${Math.max(4, safeValue * 100)}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
