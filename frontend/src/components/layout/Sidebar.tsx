import { Plus, MessageSquare, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "../ui/Button";
import { cn } from "../../lib/utils";
import { chat } from "../../api/chat";
import type { ConversationSummary } from "../../types";

interface SidebarProps {
  isOpen: boolean;
  onClose?: () => void;
  currentConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  refreshTrigger?: number; // Prop to trigger refresh from parent
}

export function Sidebar({ 
  isOpen, 
  onClose: _onClose, 
  currentConversationId, 
  onSelectConversation, 
  onNewChat,
  refreshTrigger 
}: SidebarProps) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchConversations();
  }, [refreshTrigger]);

  const fetchConversations = async () => {
    setIsLoading(true);
    try {
      const data = await chat.getConversations();
      setConversations(data);
    } catch (error) {
      console.error("Failed to load conversations", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to delete this chat?")) return;
    
    try {
      await chat.deleteConversation(id);
      if (currentConversationId === id) {
        onNewChat();
      }
      fetchConversations();
    } catch (error) {
      console.error("Failed to delete conversation", error);
    }
  };

  return (
    <aside
      className={cn(
        "fixed inset-y-0 left-0 z-50 w-64 transform bg-card border-r border-border transition-transform duration-200 ease-in-out md:relative md:translate-x-0 flex flex-col",
        !isOpen && "-translate-x-full"
      )}
    >
      <div className="flex h-full flex-col p-4">
        <div className="mb-6">
          <h2 className="mb-2 text-sm font-semibold text-muted-foreground px-2">Chat History</h2>
          <Button 
            onClick={onNewChat}
            className="w-full justify-start gap-2" 
            variant="outline"
          >
            <Plus className="h-4 w-4" />
            New Chat
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto">
          <div className="space-y-1">
            {isLoading ? (
               <div className="px-2 text-sm text-muted-foreground">Loading...</div>
            ) : conversations.length === 0 ? (
               <div className="px-2 text-sm text-muted-foreground">No recent chats</div>
            ) : (
              conversations.map((conv) => (
                <div key={conv.id} className="group relative">
                  <Button
                    variant={currentConversationId === conv.id ? "secondary" : "ghost"}
                    className="w-full justify-start text-left font-normal truncate pr-8"
                    onClick={() => onSelectConversation(conv.id)}
                  >
                    <MessageSquare className="mr-2 h-4 w-4 opacity-70" />
                    <div className="truncate">{conv.title || "Untitled Chat"}</div>
                  </Button>
                  <button
                    onClick={(e) => handleDelete(e, conv.id)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-muted-foreground opacity-0 hover:text-destructive group-hover:opacity-100 transition-opacity"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="mt-auto border-t border-border pt-4">
          <div className="px-2 text-xs text-muted-foreground">
            Logged in as User
          </div>
        </div>
      </div>
    </aside>
  );
}
