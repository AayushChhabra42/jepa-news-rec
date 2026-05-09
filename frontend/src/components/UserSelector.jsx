import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchUsers } from "../api/client";
import ErrorBanner from "./shared/ErrorBanner";
import LoadingSpinner from "./shared/LoadingSpinner";

export default function UserSelector({ selectedUserId, onSelect }) {
  const [search, setSearch] = useState("");
  const { data, isLoading, error } = useQuery({
    queryKey: ["users", search],
    queryFn: () => fetchUsers({ page: 1, pageSize: 200, search })
  });

  const users = useMemo(() => data?.users ?? [], [data]);

  return (
    <aside className="flex h-full min-h-0 flex-col gap-4 border-r border-slate-200 bg-white p-5">
      <div>
        <h1 className="text-xl font-semibold text-slate-950">JEPA News Rec</h1>
        <p className="mt-1 text-sm text-slate-500">Stage 1: context encoder retrieval</p>
      </div>
      <input
        value={search}
        onChange={(event) => setSearch(event.target.value)}
        placeholder="Search user ID"
        className="h-10 rounded-md border border-slate-300 px-3 text-sm outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100"
      />
      {isLoading && <LoadingSpinner label="Loading users" />}
      {error && <ErrorBanner error={error} />}
      <div className="min-h-0 flex-1 overflow-y-auto rounded-md border border-slate-200">
        {users.map((userId) => (
          <button
            key={userId}
            type="button"
            onClick={() => onSelect(userId)}
            className={`block w-full border-b border-slate-100 px-3 py-2 text-left text-sm last:border-b-0 ${
              selectedUserId === userId ? "bg-teal-50 font-medium text-teal-900" : "hover:bg-slate-50"
            }`}
          >
            {userId}
          </button>
        ))}
      </div>
      {data && (
        <div className="text-xs text-slate-500">
          {(data.total ?? users.length).toLocaleString()} users available
        </div>
      )}
    </aside>
  );
}
