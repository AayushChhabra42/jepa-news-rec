import axios from "axios";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api",
  timeout: 30_000
});

export async function fetchHealth() {
  const { data } = await api.get("/health");
  return data;
}

export async function fetchUsers({ page = 1, pageSize = 50, search = "" } = {}) {
  const { data } = await api.get("/users", { params: { page, page_size: pageSize } });
  if (!search) return data;
  const needle = search.toLowerCase();
  return { ...data, users: data.users.filter((user) => user.toLowerCase().includes(needle)) };
}

export async function fetchUserProfile(userId) {
  const { data } = await api.get(`/users/${encodeURIComponent(userId)}/profile`);
  return data;
}

export async function fetchRecommendations({ userId, topK = 50, stage = "jepa", labelReveal = false }) {
  const { data } = await api.get(`/users/${encodeURIComponent(userId)}/recommendations`, {
    params: { top_k: topK, stage, label_reveal: labelReveal }
  });
  return data;
}
