import { useQuery } from "@tanstack/react-query";
import { fetchUserProfile } from "../api/client";

export function useUserProfile(userId) {
  return useQuery({
    queryKey: ["userProfile", userId],
    queryFn: () => fetchUserProfile(userId),
    enabled: Boolean(userId)
  });
}
