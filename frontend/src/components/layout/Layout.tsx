

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      {/* Sidebar is now managed by the page (e.g. ChatPage) */}
      
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Mobile header logic should be handled by the page if it controls the sidebar */}
        <main className="flex-1 overflow-hidden relative">
          {children}
        </main>
      </div>
    </div>
  );
}
