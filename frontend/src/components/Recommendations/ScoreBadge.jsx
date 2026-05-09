function colorFor(value) {
  if (value == null) return "bg-slate-100 text-slate-500";
  if (value > 0.7) return "bg-emerald-50 text-emerald-700";
  if (value > 0.4) return "bg-amber-50 text-amber-700";
  return "bg-red-50 text-red-700";
}

export default function ScoreBadge({ label, value }) {
  return (
    <span className={`inline-flex items-center gap-2 rounded px-2 py-1 text-xs font-medium ${colorFor(value)}`}>
      <span>{label}</span>
      <span>{value == null ? "—" : value.toFixed(3)}</span>
    </span>
  );
}
