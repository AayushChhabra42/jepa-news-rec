export default function RevealButton({ revealed, onToggle, disabled }) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onToggle}
      className="h-9 rounded-md bg-slate-950 px-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-300"
    >
      {revealed ? "Hide labels" : "Reveal labels"}
    </button>
  );
}
