/**
 * Lightweight markdown rendering for assistant messages (no extra deps).
 */

function inlineFormat(text) {
  const parts = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let match;

  while ((match = re.exec(text)) !== null) {
    if (match.index > last) {
      parts.push({ type: "text", value: text.slice(last, match.index) });
    }
    const token = match[0];
    if (token.startsWith("**")) {
      parts.push({ type: "bold", value: token.slice(2, -2) });
    } else if (token.startsWith("`")) {
      parts.push({ type: "code", value: token.slice(1, -1) });
    } else if (token.startsWith("*")) {
      parts.push({ type: "italic", value: token.slice(1, -1) });
    } else {
      const linkMatch = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(token);
      if (linkMatch) {
        parts.push({ type: "link", label: linkMatch[1], href: linkMatch[2] });
      }
    }
    last = match.index + token.length;
  }

  if (last < text.length) {
    parts.push({ type: "text", value: text.slice(last) });
  }

  return parts.length ? parts : [{ type: "text", value: text }];
}

function renderInline(text, keyPrefix) {
  return inlineFormat(text).map((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.type === "bold") {
      return (
        <strong key={key} className="font-semibold text-inherit">
          {part.value}
        </strong>
      );
    }
    if (part.type === "italic") {
      return (
        <em key={key} className="italic text-inherit">
          {part.value}
        </em>
      );
    }
    if (part.type === "code") {
      return (
        <code
          key={key}
          className="rounded-md bg-slate-900/10 px-1.5 py-0.5 font-mono text-[0.9em] dark:bg-white/10"
        >
          {part.value}
        </code>
      );
    }
    if (part.type === "link") {
      return (
        <a
          key={key}
          href={part.href}
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium text-cyan-600 underline decoration-cyan-600/40 underline-offset-2 hover:text-cyan-500 dark:text-cyan-400"
        >
          {part.label}
        </a>
      );
    }
    return <span key={key}>{part.value}</span>;
  });
}

export function MarkdownContent({ text, className = "" }) {
  if (!text) return null;

  const lines = text.split("\n");
  const blocks = [];
  let i = 0;
  let blockKey = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.startsWith("```")) {
      const codeLines = [];
      i += 1;
      while (i < lines.length && !lines[i].startsWith("```")) {
        codeLines.push(lines[i]);
        i += 1;
      }
      i += 1;
      blocks.push(
        <pre
          key={`block-${blockKey++}`}
          className="my-2 overflow-x-auto rounded-xl border border-slate-200/80 bg-slate-900/5 p-3 font-mono text-[13px] leading-relaxed dark:border-slate-700 dark:bg-black/30"
        >
          <code>{codeLines.join("\n")}</code>
        </pre>
      );
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      blocks.push(
        <p
          key={`block-${blockKey++}`}
          className="mt-2 mb-1 text-[15px] font-semibold leading-snug text-inherit"
        >
          {renderInline(heading[2], `h-${blockKey}`)}
        </p>
      );
      i += 1;
      continue;
    }

    if (/^[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^[-*]\s+/, ""));
        i += 1;
      }
      blocks.push(
        <ul key={`block-${blockKey++}`} className="my-2 list-disc space-y-1 pl-5">
          {items.map((item, j) => (
            <li key={j}>{renderInline(item, `li-${blockKey}-${j}`)}</li>
          ))}
        </ul>
      );
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\d+\.\s+/, ""));
        i += 1;
      }
      blocks.push(
        <ol key={`block-${blockKey++}`} className="my-2 list-decimal space-y-1 pl-5">
          {items.map((item, j) => (
            <li key={j}>{renderInline(item, `oli-${blockKey}-${j}`)}</li>
          ))}
        </ol>
      );
      continue;
    }

    if (line.trim() === "") {
      i += 1;
      continue;
    }

    const paraLines = [];
    while (i < lines.length && lines[i].trim() !== "" && !lines[i].startsWith("```")) {
      paraLines.push(lines[i]);
      i += 1;
    }
    blocks.push(
      <p key={`block-${blockKey++}`} className="whitespace-pre-wrap break-words">
        {paraLines.map((pl, j) => (
          <span key={j}>
            {j > 0 ? <br /> : null}
            {renderInline(pl, `p-${blockKey}-${j}`)}
          </span>
        ))}
      </p>
    );
  }

  const rootClass = className ? `space-y-1 ${className}` : "space-y-1";
  return <div className={rootClass}>{blocks}</div>;
}


