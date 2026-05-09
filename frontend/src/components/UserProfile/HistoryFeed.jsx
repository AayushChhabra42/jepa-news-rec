const categoryClasses = {
  news: "bg-blue-50 text-blue-700",
  sports: "bg-emerald-50 text-emerald-700",
  finance: "bg-amber-50 text-amber-700",
  lifestyle: "bg-rose-50 text-rose-700",
  travel: "bg-cyan-50 text-cyan-700",
  video: "bg-violet-50 text-violet-700"
};

export default function HistoryFeed({ history = [] }) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto rounded-md border border-slate-200 bg-white">
      <div className="sticky top-0 border-b border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-800">
        Recent click history
      </div>
      {history.length === 0 ? (
        <div className="p-4 text-sm text-slate-500">No click history for this user.</div>
      ) : (
        <div className="divide-y divide-slate-100">
          {history.slice().reverse().map((article) => (
            <article key={`${article.article_id}-${article.title}`} className="p-4">
              <div className="line-clamp-2 text-sm font-medium text-slate-900">{article.title}</div>
              <div className="mt-2 flex flex-wrap gap-2 text-xs">
                <span className={`rounded px-2 py-1 ${categoryClasses[article.category] || "bg-slate-100 text-slate-700"}`}>
                  {article.category}
                </span>
                <span className="rounded bg-slate-100 px-2 py-1 text-slate-600">{article.subcategory}</span>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
