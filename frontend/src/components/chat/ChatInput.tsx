import { useRef, useEffect, useState, useCallback } from "react";
import { SendHorizontal, Paperclip, X, FileText, FileSpreadsheet } from "lucide-react";
import { Button } from "../ui/Button";
import { Textarea } from "../ui/Textarea";
import type { ChatAttachment } from "../../types";

const MAX_FILES = 3;
const MAX_SIZE_MB = 10;
const ACCEPTED_TYPES = "image/*,.pdf,.docx,.csv,.xlsx";

interface ChatInputProps {
  onSend: (message: string, files?: File[]) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const canSend = (input.trim() || attachments.length > 0) && !disabled;

  const classifyFile = (file: File): ChatAttachment['type'] => {
    if (file.type.startsWith('image/')) return 'image';
    if (file.name.match(/\.(csv|xlsx)$/i)) return 'spreadsheet';
    return 'document';
  };

  const addFiles = useCallback((fileList: FileList | File[]) => {
    const files = Array.from(fileList);
    setError(null);

    setAttachments(prev => {
      const total = prev.length + files.length;
      if (total > MAX_FILES) {
        setError(`Maximum ${MAX_FILES} files allowed.`);
        return prev;
      }

      const newAttachments: ChatAttachment[] = [];
      for (const file of files) {
        if (file.size > MAX_SIZE_MB * 1024 * 1024) {
          setError(`${file.name} is too large (max ${MAX_SIZE_MB}MB).`);
          continue;
        }
        newAttachments.push({
          id: crypto.randomUUID(),
          file,
          preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined,
          type: classifyFile(file),
        });
      }
      return [...prev, ...newAttachments];
    });
  }, []);

  const removeAttachment = (id: string) => {
    setAttachments(prev => {
      const att = prev.find(a => a.id === id);
      if (att?.preview) URL.revokeObjectURL(att.preview);
      return prev.filter(a => a.id !== id);
    });
    setError(null);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addFiles(e.target.files);
    e.target.value = "";
  };

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!canSend) return;
    const files = attachments.length > 0 ? attachments.map(a => a.file) : undefined;
    onSend(input, files);
    // Cleanup previews
    attachments.forEach(a => { if (a.preview) URL.revokeObjectURL(a.preview); });
    setInput("");
    setAttachments([]);
    setError(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
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
        <form
          onSubmit={handleSubmit}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`relative flex flex-col rounded-xl bg-muted/50 p-2 focus-within:ring-1 focus-within:ring-ring transition-all ${
            isDragOver ? "ring-2 ring-primary border-dashed border-2 border-primary/50" : ""
          }`}
        >
          {/* Attachment preview strip */}
          {attachments.length > 0 && (
            <div className="flex flex-wrap gap-2 px-2 pt-1 pb-2">
              {attachments.map(att => (
                <div
                  key={att.id}
                  className="flex items-center gap-1.5 rounded-lg bg-background border border-border px-2 py-1.5 text-xs shadow-sm"
                >
                  {att.type === 'image' && att.preview ? (
                    <img src={att.preview} alt={att.file.name} className="h-8 w-8 rounded object-cover" />
                  ) : att.type === 'spreadsheet' ? (
                    <FileSpreadsheet className="h-4 w-4 text-green-600 shrink-0" />
                  ) : (
                    <FileText className="h-4 w-4 text-blue-600 shrink-0" />
                  )}
                  <span className="max-w-[120px] truncate text-muted-foreground">
                    {att.file.name}
                  </span>
                  <button
                    type="button"
                    onClick={() => removeAttachment(att.id)}
                    className="ml-0.5 rounded-full p-0.5 hover:bg-muted transition-colors"
                  >
                    <X className="h-3 w-3 text-muted-foreground" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Input row */}
          <div className="flex items-end gap-2">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={ACCEPTED_TYPES}
              onChange={handleFileSelect}
              className="hidden"
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled || attachments.length >= MAX_FILES}
              className="h-10 w-10 shrink-0 mb-1 text-muted-foreground hover:text-foreground"
              title="Attach files"
            >
              <Paperclip className="h-5 w-5" />
            </Button>

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
              disabled={!canSend}
              className="h-10 w-10 shrink-0 mb-1 mr-1 bg-orange-500 hover:bg-orange-600 rounded-lg transition-colors"
            >
              <SendHorizontal className="h-5 w-5 text-white" />
            </Button>
          </div>

          {/* Error message */}
          {error && (
            <div className="px-3 pb-1 text-xs text-red-500">{error}</div>
          )}
        </form>

        {isDragOver && (
          <div className="absolute inset-0 flex items-center justify-center rounded-xl bg-primary/5 pointer-events-none">
            <p className="text-sm font-medium text-primary">Drop files here</p>
          </div>
        )}

        <div className="mt-2 text-center text-xs text-muted-foreground">
          AI can make mistakes. Please verify important information.
        </div>
      </div>
    </div>
  );
}
