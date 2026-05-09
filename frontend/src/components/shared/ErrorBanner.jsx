export default function ErrorBanner({ error }) {
  const message = error?.response?.data?.detail || error?.message || "Something went wrong.";
  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
      {message}
    </div>
  );
}
