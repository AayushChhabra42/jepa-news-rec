import { useState } from "react";
import { useRecommendations } from "../../hooks/useRecommendations";
import ErrorBanner from "../shared/ErrorBanner";
import LoadingSpinner from "../shared/LoadingSpinner";
import ArticleCard from "./ArticleCard";
import MetricStrip from "./MetricStrip";
import RankDeltaView from "./RankDeltaView";
import RevealButton from "./RevealButton";
import StageToggle from "./StageToggle";

export default function RecommendationPanel({ userId }) {
  const [labelReveal, setLabelReveal] = useState(false);
  const [topK, setTopK] = useState(50);
  const [stage, setStage] = useState("jepa");
  const [viewMode, setViewMode] = useState("list");
  const { data, isLoading, error, isFetching } = useRecommendations({
    userId,
    topK,
    stage,
    labelReveal
  });

  if (!userId) return <section className="p-5 text-sm text-slate-500">Recommendations appear after user selection.</section>;

  return (
    <section className="flex h-full min-h-0 flex-col gap-4 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-500">Recommendations</div>
          <h2 className="mt-1 text-lg font-semibold text-slate-950">
            {stage === "jepa" ? "JEPA ranked candidates" : "JEPA candidates reranked by XGBoost"}
          </h2>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <StageToggle stage={stage} onStageChange={setStage} />
          <select
            value={topK}
            onChange={(event) => setTopK(Number(event.target.value))}
            className="h-9 rounded-md border border-slate-300 bg-white px-2 text-sm"
          >
            {[25, 50, 100].map((value) => (
              <option key={value} value={value}>Top {value}</option>
            ))}
          </select>
          <RevealButton revealed={labelReveal} disabled={isFetching} onToggle={() => setLabelReveal((value) => !value)} />
        </div>
      </div>
      {stage === "both" && (
        <div className="flex justify-end">
          <div className="grid h-8 grid-cols-2 rounded-md border border-slate-300 bg-slate-100 p-0.5 text-xs font-medium">
            {[
              ["list", "List"],
              ["delta", "Rank delta"]
            ].map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => setViewMode(value)}
                className={`rounded px-3 ${viewMode === value ? "bg-white text-slate-950 shadow-sm" : "text-slate-500 hover:text-slate-800"}`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}
      {isLoading && <LoadingSpinner label="Scoring candidates" />}
      {error && <ErrorBanner error={error} />}
      {data && (
        <>
          <MetricStrip metrics={data.metrics} />
          <div className="min-h-0 flex-1 overflow-y-auto pr-1">
            {stage === "both" && viewMode === "delta" ? (
              <div className="overflow-x-auto">
                <RankDeltaView recommendations={data.recommendations} />
              </div>
            ) : (
              <div className="grid gap-3">
                {data.recommendations.map((article) => (
                  <ArticleCard key={article.article_id} article={article} labelReveal={labelReveal} />
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </section>
  );
}
