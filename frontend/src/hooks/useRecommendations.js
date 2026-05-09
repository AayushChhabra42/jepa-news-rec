import { useQuery } from "@tanstack/react-query";
import { fetchRecommendations } from "../api/client";

export function useRecommendations({ userId, topK = 50, stage = "jepa", labelReveal = false }) {
  return useQuery({
    queryKey: ["recommendations", userId, topK, stage, labelReveal],
    queryFn: () => fetchRecommendations({ userId, topK, stage, labelReveal }),
    enabled: Boolean(userId)
  });
}
