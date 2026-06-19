/**
 * Contextual workflow chips attached to the latest bot message (Phase 3).
 */
export default function WorkflowActionBar({ actions, onAction, disabled }) {
  if (!actions?.length) return null;

  return (
    <div className="workflow-actions" role="group" aria-label="Suggested replies">
      {actions.map((action) => {
        const label = action.label_bn || action.label || action.message;
        const kind = action.kind || "secondary";
        return (
          <button
            key={action.id}
            type="button"
            className={`workflow-action-btn workflow-action-${kind}`}
            disabled={disabled}
            onClick={() => onAction?.(action)}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
