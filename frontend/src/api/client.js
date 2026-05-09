import axios from "axios";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api",
  timeout: 30_000,
  headers: {
    "ngrok-skip-browser-warning": "true"
  }
});

export const apiBaseUrl = api.defaults.baseURL;

export async function fetchHealth() {
  const { data } = await api.get("/health");
  return data;
}

export async function fetchUsers({ page = 1, pageSize = 50, search = "" } = {}) {
  const { data } = await api.get("/users", { params: { page, page_size: pageSize } });
  if (!Array.isArray(data) && (typeof data !== "object" || data === null || !Array.isArray(data.users))) {
    const preview = typeof data === "string" ? data.slice(0, 120) : JSON.stringify(data).slice(0, 120);
    throw new Error(`Unexpected /users response from API. Check VITE_API_BASE_URL. Response starts: ${preview}`);
  }
  const payload = Array.isArray(data)
    ? { users: data, page, page_size: pageSize, total: data.length }
    : { users: data.users ?? [], page: data.page ?? page, page_size: data.page_size ?? pageSize, total: data.total ?? data.users?.length ?? 0 };
  if (!search) return payload;
  const needle = search.toLowerCase();
  const users = payload.users.filter((user) => String(user).toLowerCase().includes(needle));
  return { ...payload, users, total: payload.total || users.length };
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
