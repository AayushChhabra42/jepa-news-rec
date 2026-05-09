import { useUserProfile } from "../../hooks/useUserProfile";
import ErrorBanner from "../shared/ErrorBanner";
import LoadingSpinner from "../shared/LoadingSpinner";
import CategoryBarChart from "./CategoryBarChart";
import HistoryFeed from "./HistoryFeed";
import StatStrip from "./StatStrip";

export default function UserProfile({ userId }) {
  const { data, isLoading, error } = useUserProfile(userId);

  if (!userId) return <section className="p-5 text-sm text-slate-500">Select a user to inspect their profile.</section>;
  if (isLoading) return <section className="p-5"><LoadingSpinner label="Loading profile" /></section>;
  if (error) return <section className="p-5"><ErrorBanner error={error} /></section>;

  return (
    <section className="flex h-full min-h-0 flex-col gap-4 p-5">
      <div>
        <div className="text-xs uppercase tracking-wide text-slate-500">User profile</div>
        <h2 className="mt-1 text-lg font-semibold text-slate-950">{data.user_id}</h2>
      </div>
      <StatStrip features={data.features} />
      <CategoryBarChart categories={data.features.top_categories} />
      <HistoryFeed history={data.history} />
    </section>
  );
}
