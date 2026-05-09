const stages = [
  { label: "JEPA only", value: "jepa" },
  { label: "JEPA + XGBoost", value: "both" }
];

export default function StageToggle({ stage, onStageChange }) {
  return (
    <div className="grid h-9 grid-cols-2 rounded-md border border-slate-300 bg-slate-100 p-0.5 text-xs font-medium">
      {stages.map((item) => (
        <button
          key={item.value}
          type="button"
          onClick={() => onStageChange(item.value)}
          className={`rounded px-3 transition ${
            stage === item.value ? "bg-white text-slate-950 shadow-sm" : "text-slate-500 hover:text-slate-800"
          }`}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
