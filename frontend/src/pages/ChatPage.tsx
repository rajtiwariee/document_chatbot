import { useState } from "react";

import { Sidebar } from "../components/layout/Sidebar";
import { ChatInterface } from "../components/chat/ChatInterface";

export function ChatPage() {
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleConversationCreated = (id: string) => {
    setCurrentConversationId(id);
    setRefreshTrigger(prev => prev + 1); // Refresh sidebar
  };

  const handleSelectConversation = (id: string) => {
    setCurrentConversationId(id);
    setSidebarOpen(false);
  };

  const handleNewChat = () => {
    setCurrentConversationId(null);
    setSidebarOpen(false);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      <Sidebar 
        isOpen={sidebarOpen} 
        onClose={() => setSidebarOpen(false)}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        refreshTrigger={refreshTrigger}
      />
      
      <div className="flex flex-1 flex-col overflow-hidden">
         {/* Mobile Header logic was in Layout, but now we're composing manually. 
             Ideally Layout handles this, but since we pulled state up, we need to pass props to Layout or compose here.
             Let's reuse Layout but we need to modify Layout to accept Sidebar props or children.
             Actually, Layout currently hardcodes Sidebar. Let's refactor Layout or just inline the structure here since ChatPage controls the state now.
             Given the previous Layout code, it had local state. We should modify Layout to accept external control or just copy the structure here since it's the main page.
             Let's use the inline structure for now to ensure state connectivity without rewriting Layout completely yet.
         */}
         <header className="flex h-14 items-center gap-4 border-b bg-card px-6 md:hidden">
            <button
              className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground h-10 w-10"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5"><line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="18" y2="18"/></svg>
            </button>
            <span className="font-semibold">Team Docs</span>
         </header>

         <main className="flex-1 overflow-hidden relative">
            <ChatInterface 
              conversationId={currentConversationId}
              onConversationCreated={handleConversationCreated}
            />
         </main>
      </div>

      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  );
}
