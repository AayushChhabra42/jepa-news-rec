import { useState } from "react";
import UserSelector from "./components/UserSelector";
import UserProfile from "./components/UserProfile/UserProfile";
import RecommendationPanel from "./components/Recommendations/RecommendationPanel";

export default function App() {
  const [selectedUserId, setSelectedUserId] = useState("");

  return (
    <main className="grid h-screen min-h-0 grid-cols-1 overflow-hidden lg:grid-cols-[280px_minmax(320px,420px)_1fr]">
      <UserSelector selectedUserId={selectedUserId} onSelect={setSelectedUserId} />
      <div className="min-h-0 overflow-hidden border-r border-slate-200">
        <UserProfile userId={selectedUserId} />
      </div>
      <div className="min-h-0 overflow-hidden">
        <RecommendationPanel userId={selectedUserId} />
      </div>
    </main>
  );
}
