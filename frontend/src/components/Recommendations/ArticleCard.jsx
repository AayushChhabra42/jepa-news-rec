const categoryClasses = {
  news: "bg-blue-50 text-blue-700",
  sports: "bg-emerald-50 text-emerald-700",
  finance: "bg-amber-50 text-amber-700",
  lifestyle: "bg-rose-50 text-rose-700",
  travel: "bg-cyan-50 text-cyan-700",
  video: "bg-violet-50 text-violet-700",
  health: "bg-lime-50 text-lime-700",
  foodanddrink: "bg-orange-50 text-orange-700"
};

function scoreColor(score) {
  if (score > 0.7) return "bg-emerald-500";
  if (score > 0.4) return "bg-amber-500";
  return "bg-red-500";
}

function rankDeltaLabel(delta) {
  if (delta == null) return null;
  if (delta > 0) return { text: `up ${delta}`, className: "bg-emerald-50 text-emerald-700" };
  if (delta < 0) return { text: `down ${Math.abs(delta)}`, className: "bg-red-50 text-red-700" };
  return { text: "=", className: "bg-slate-100 text-slate-600" };
}

export default function ArticleCard({ article, labelReveal }) {
  const normalized = Math.max(0, Math.min(1, (article.jepa_score + 1) / 2));
  const rank = article.final_rank ?? article.rank;
  const delta = rankDeltaLabel(article.rank_delta);

  return (
    <article className="relative rounded-md border border-slate-200 bg-white p-4 pl-12 shadow-sm">
      <div className="absolute left-3 top-3 flex h-7 w-7 items-center justify-center rounded bg-slate-950 text-xs font-semibold text-white">
        {rank}
      </div>
      <div className="line-clamp-2 min-h-10 text-sm font-semibold leading-5 text-slate-950">{article.title}</div>
      {(article.jepa_rank || article.xgb_rank) && (
        <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-500">
          {article.jepa_rank && <span>JEPA #{article.jepa_rank}</span>}
          {article.xgb_rank && <span>XGB #{article.xgb_rank}</span>}
          {delta && <span className={`rounded px-2 py-0.5 font-semibold ${delta.className}`}>{delta.text}</span>}
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-2 text-xs">
        <span className={`rounded px-2 py-1 ${categoryClasses[article.category] || "bg-slate-100 text-slate-700"}`}>
          {article.category}
        </span>
        <span className="rounded bg-slate-100 px-2 py-1 text-slate-600">{article.subcategory}</span>
      </div>
      {article.abstract && <p className="mt-3 line-clamp-2 text-xs leading-5 text-slate-500">{article.abstract}</p>}
      <div className="mt-4 grid gap-3">
        <div>
          <div className="mb-1 flex justify-between text-xs text-slate-500">
            <span>JEPA score</span>
            <span>{article.jepa_score.toFixed(3)}</span>
          </div>
          <div className="h-2 rounded bg-slate-100">
            <div className={`h-full rounded ${scoreColor(normalized)}`} style={{ width: `${normalized * 100}%` }} />
          </div>
        </div>
        <div className="flex items-center justify-between gap-3 text-xs">
          <span className="rounded bg-slate-100 px-2 py-1 text-slate-500">
            XGB score: {article.xgb_score == null ? "-- Stage 2" : article.xgb_score.toFixed(3)}
          </span>
          <span className={`rounded px-2 py-1 font-semibold ${labelReveal ? (article.label === 1 ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700") : "bg-slate-100 text-slate-400 blur-[2px]"}`}>
            {labelReveal ? (article.label === 1 ? "clicked" : "not clicked") : "hidden"}
          </span>
        </div>
      </div>
    </article>
  );
}
