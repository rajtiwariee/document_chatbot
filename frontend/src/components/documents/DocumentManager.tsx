import { useState, useEffect, useRef } from 'react';
import { Upload, FileText, Trash2, X, AlertCircle, Loader2, FolderOpen } from 'lucide-react';
import { Button } from '../ui/Button';
import { cn } from '../../lib/utils';
import { documents } from '../../api/documents';
import type { Document } from '../../types';

interface DocumentManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

export function DocumentManager({ isOpen, onClose }: DocumentManagerProps) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      fetchDocuments();
    }
  }, [isOpen]);

  const fetchDocuments = async () => {
    setIsLoading(true);
    try {
      // The API client returns Document[], not the raw response
      const data = await documents.getAll();
      setDocs(data);
    } catch (error) {
      console.error("Failed to load documents", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadError(null);

    try {
      await documents.upload(file);
      // Refresh list after upload
      fetchDocuments();
    } catch (error: any) {
      console.error("Upload failed", error);
      setUploadError(error.response?.data?.detail || "Failed to upload document");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this document? This cannot be undone.")) return;
    
    try {
      await documents.delete(id);
      setDocs(prev => prev.filter(d => d.id !== id));
    } catch (error) {
      console.error("Failed to delete document", error);
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 animate-in fade-in duration-200">
      <div className="bg-card w-full max-w-2xl rounded-lg border border-border shadow-xl flex flex-col max-h-[80vh]">
        
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center gap-2">
            <FolderOpen className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">Document Manager</h2>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden flex flex-col p-4 gap-4">
          
          {/* Upload Area */}
          <div 
            className={cn(
              "border-2 border-dashed border-border rounded-lg p-8 flex flex-col items-center justify-center text-center transition-colors cursor-pointer hover:bg-muted/50 hover:border-primary/50",
              isUploading && "opacity-50 pointer-events-none"
            )}
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              className="hidden" 
              ref={fileInputRef} 
              onChange={handleFileSelect}
              accept=".pdf,.docx,.txt,.md"
            />
            
            {isUploading ? (
              <div className="flex flex-col items-center gap-2">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
                <p className="text-sm text-muted-foreground">Uploading and processing...</p>
              </div>
            ) : (
              <>
                <Upload className="h-8 w-8 text-muted-foreground mb-2" />
                <p className="font-medium">Click to upload document</p>
                <p className="text-xs text-muted-foreground mt-1">PDF, DOCX, TXT (Max 10MB)</p>
              </>
            )}
          </div>

          {uploadError && (
            <div className="bg-destructive/10 text-destructive text-sm p-3 rounded-md flex items-center gap-2">
              <AlertCircle className="h-4 w-4" />
              {uploadError}
            </div>
          )}

          {/* Document List */}
          <div className="flex-1 overflow-y-auto min-h-[200px] border border-border rounded-md">
            {isLoading ? (
               <div className="flex items-center justify-center h-full text-muted-foreground text-sm">Loading...</div>
            ) : docs.length === 0 ? (
               <div className="flex flex-col items-center justify-center h-full text-muted-foreground text-sm gap-2">
                 <FileText className="h-8 w-8 opacity-20" />
                 No documents uploaded yet
               </div>
            ) : (
               <div className="divide-y divide-border">
                 {docs.map(doc => (
                   <div key={doc.id} className="flex items-center justify-between p-3 hover:bg-muted/30 transition-colors">
                     <div className="flex items-center gap-3 min-w-0">
                       <div className="bg-primary/10 p-2 rounded">
                         <FileText className="h-4 w-4 text-primary" />
                       </div>
                       <div className="flex-col min-w-0">
                         <p className="text-sm font-medium truncate">{doc.filename}</p>
                         <p className="text-xs text-muted-foreground">
                           {formatSize(doc.file_size)} • {new Date(doc.created_at).toLocaleDateString()}
                         </p>
                       </div>
                     </div>
                     
                     <div className="flex items-center gap-3">
                        <span className={cn(
                          "text-xs px-2 py-0.5 rounded-full capitalize",
                          doc.status === 'completed' && "bg-green-500/10 text-green-500",
                          doc.status === 'processing' && "bg-blue-500/10 text-blue-500",
                          doc.status === 'failed' && "bg-red-500/10 text-red-500",
                          doc.status === 'pending' && "bg-yellow-500/10 text-yellow-500",
                        )}>
                          {doc.status}
                        </span>
                        
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          className="h-8 w-8 text-muted-foreground hover:text-destructive"
                          onClick={() => handleDelete(doc.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                     </div>
                   </div>
                 ))}
               </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}
