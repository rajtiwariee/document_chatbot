import { useRef, useEffect, useState } from "react";
import { SendHorizontal } from "lucide-react";
import { Button } from "../ui/Button";
import { Textarea } from "../ui/Textarea";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || disabled) return;
    onSend(input);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  return (
    <div className="p-4 bg-background border-t border-border">
      <div className="mx-auto max-w-3xl relative">
        <form onSubmit={handleSubmit} className="relative flex items-end gap-2 rounded-xl bg-muted/50 p-2 focus-within:ring-1 focus-within:ring-ring">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about team projects, policies, or docs..."
            className="min-h-[50px] w-full resize-none border-0 bg-transparent px-3 py-3 focus-visible:ring-0 shadow-none"
            disabled={disabled}
          />
          <Button 
            type="submit" 
            size="icon" 
            disabled={!input.trim() || disabled}
            className="h-10 w-10 shrink-0 mb-1 mr-1 bg-orange-500 hover:bg-orange-600 rounded-lg transition-colors"
          >
            <SendHorizontal className="h-5 w-5 text-white" />
          </Button>
        </form>
        <div className="mt-2 text-center text-xs text-muted-foreground">
          AI can make mistakes. Please verify important information.
        </div>
      </div>
    </div>
  );
}
