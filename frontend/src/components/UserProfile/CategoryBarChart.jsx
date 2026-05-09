import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function CategoryBarChart({ categories = [] }) {
  const data = categories.map((item) => ({ ...item, pctLabel: Math.round(item.pct * 100) }));
  return (
    <div className="h-56 rounded-md border border-slate-200 bg-white p-3">
      <div className="mb-2 text-sm font-semibold text-slate-800">Top categories</div>
      <ResponsiveContainer width="100%" height="85%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 24, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" hide domain={[0, 100]} />
          <YAxis dataKey="category" type="category" width={84} tick={{ fontSize: 11 }} />
          <Tooltip formatter={(value) => [`${value}%`, "Share"]} />
          <Bar dataKey="pctLabel" fill="#0f766e" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
