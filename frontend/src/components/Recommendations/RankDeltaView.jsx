function formatScore(score) {
  return score == null ? "--" : score.toFixed(3);
}

function DeltaBadge({ value }) {
  if (value > 0) return <span className="rounded bg-emerald-50 px-2 py-1 text-emerald-700">up {value}</span>;
  if (value < 0) return <span className="rounded bg-red-50 px-2 py-1 text-red-700">down {Math.abs(value)}</span>;
  return <span className="rounded bg-slate-100 px-2 py-1 text-slate-600">=</span>;
}

function RankColumn({ title, recommendations, rankKey }) {
  const sorted = [...recommendations].sort((a, b) => (a[rankKey] ?? a.rank) - (b[rankKey] ?? b.rank));
  return (
    <div className="min-w-0">
      <div className="sticky top-0 z-10 border-b border-slate-200 bg-slate-50 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        {title}
      </div>
      <div className="divide-y divide-slate-200">
        {sorted.map((article) => (
          <div key={`${title}-${article.article_id}`} className="grid grid-cols-[2.25rem_1fr_auto] gap-3 bg-white p-3 text-xs">
            <div className="flex h-7 w-7 items-center justify-center rounded bg-slate-950 font-semibold text-white">
              {article[rankKey] ?? article.rank}
            </div>
            <div className="min-w-0">
              <div className="line-clamp-2 font-semibold leading-5 text-slate-950">{article.title}</div>
              <div className="mt-1 flex flex-wrap gap-2 text-slate-500">
                <span>{article.category}</span>
                <span>{article.subcategory}</span>
              </div>
              <div className="mt-2 flex flex-wrap gap-2 text-slate-500">
                <span>JEPA {formatScore(article.jepa_score)}</span>
                <span>XGB {formatScore(article.xgb_score)}</span>
              </div>
            </div>
            <div className="self-start">
              <DeltaBadge value={article.rank_delta ?? 0} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function RankDeltaView({ recommendations }) {
  return (
    <div className="grid min-w-[720px] grid-cols-2 overflow-hidden rounded-md border border-slate-200">
      <RankColumn title="JEPA rank" recommendations={recommendations} rankKey="jepa_rank" />
      <RankColumn title="XGBoost rank" recommendations={recommendations} rankKey="xgb_rank" />
    </div>
  );
}
