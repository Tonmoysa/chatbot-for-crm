import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import WorkflowActionBar from "./WorkflowActionBar.jsx";

describe("WorkflowActionBar", () => {
  it("renders nothing when actions empty", () => {
    const { container } = render(<WorkflowActionBar actions={[]} onAction={vi.fn()} />);
    expect(container.firstChild).toBeNull();
  });

  it("fires onAction with chip payload", () => {
    const onAction = vi.fn();
    const actions = [
      { id: "yes", label: "Yes", label_bn: "হ্যাঁ", message: "yes", kind: "primary" },
    ];
    render(<WorkflowActionBar actions={actions} onAction={onAction} />);
    fireEvent.click(screen.getByRole("button", { name: "হ্যাঁ" }));
    expect(onAction).toHaveBeenCalledWith(actions[0]);
  });
});
