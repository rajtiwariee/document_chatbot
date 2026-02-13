import { useState, useEffect, useRef, useMemo } from 'react';
import { Upload, FileText, Trash2, X, AlertCircle, Loader2, FolderOpen, Search, ChevronLeft, ChevronRight, ArrowUpDown } from 'lucide-react';
import { Button } from '../ui/Button';
import { cn } from '../../lib/utils';
import { documents } from '../../api/documents';
import type { Document } from '../../types';

interface DocumentManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

type SortField = 'filename' | 'created_at' | 'file_size';
type SortOrder = 'asc' | 'desc';

export function DocumentManager({ isOpen, onClose }: DocumentManagerProps) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  
  // Search & Sort State
  const [searchTerm, setSearchTerm] = useState('');
  const [sortField, setSortField] = useState<SortField>('created_at');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  
  // Pagination State
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 7;

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      fetchDocuments();
      setSearchTerm('');
      setCurrentPage(1);
    }
  }, [isOpen]);

  const fetchDocuments = async () => {
    setIsLoading(true);
    try {
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

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc'); // Default to desc for new field
    }
  };

  // Filter and Sort Logic
  const filteredAndSortedDocs = useMemo(() => {
    let result = [...docs];

    // Filter
    if (searchTerm) {
      const lowerTerm = searchTerm.toLowerCase();
      result = result.filter(doc => 
        doc.filename.toLowerCase().includes(lowerTerm)
      );
    }

    // Sort
    result.sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case 'filename':
          comparison = a.filename.localeCompare(b.filename);
          break;
        case 'created_at':
          comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
          break;
        case 'file_size':
          comparison = a.file_size - b.file_size;
          break;
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return result;
  }, [docs, searchTerm, sortField, sortOrder]);

  // Pagination Logic
  const totalPages = Math.max(1, Math.ceil(filteredAndSortedDocs.length / itemsPerPage));
  const paginatedDocs = filteredAndSortedDocs.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 animate-in fade-in duration-200">
      <div className="bg-card w-full max-w-4xl rounded-xl border border-border shadow-2xl flex flex-col max-h-[85vh] overflow-hidden">
        
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border bg-muted/20">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <FolderOpen className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h2 className="text-xl font-semibold">Document Manager</h2>
              <p className="text-sm text-muted-foreground">{docs.length} documents uploaded</p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} className="rounded-full hover:bg-gray-200/50 dark:hover:bg-gray-700/50">
            <X className="h-6 w-6 opacity-70" />
          </Button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden flex flex-col p-6 gap-6">
          
          {/* Top Bar: Search & Upload */}
          <div className="flex flex-col sm:flex-row gap-4 justify-between items-start sm:items-center">
            <div className="relative w-full sm:w-72">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search documents..."
                value={searchTerm}
                onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }}
                className="w-full pl-9 pr-4 py-2 text-sm bg-background border border-input rounded-md focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>

            <Button onClick={() => fileInputRef.current?.click()} className="whitespace-nowrap gap-2">
              {isUploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
              Upload Document
            </Button>
            <input 
              type="file" 
              className="hidden" 
              ref={fileInputRef} 
              onChange={handleFileSelect}
              accept=".pdf,.docx,.txt,.md,.csv,.xlsx,.pptx"
            />
          </div>

          {uploadError && (
            <div className="bg-destructive/10 text-destructive text-sm p-3 rounded-md flex items-center gap-2">
              <AlertCircle className="h-4 w-4" />
              {uploadError}
            </div>
          )}

          {/* Table */}
          <div className="flex-1 overflow-auto border border-border rounded-lg shadow-sm">
            <table className="w-full text-sm text-left">
              <thead className="bg-muted/50 text-muted-foreground font-medium sticky top-0 z-10">
                <tr>
                  <th className="px-4 py-3 cursor-pointer hover:bg-muted/80 transition-colors" onClick={() => handleSort('filename')}>
                    <div className="flex items-center gap-1">Name <ArrowUpDown className="h-3 w-3" /></div>
                  </th>
                  <th className="px-4 py-3 cursor-pointer hover:bg-muted/80 transition-colors w-32" onClick={() => handleSort('created_at')}>
                     <div className="flex items-center gap-1">Date <ArrowUpDown className="h-3 w-3" /></div>
                  </th>
                  <th className="px-4 py-3 cursor-pointer hover:bg-muted/80 transition-colors w-28" onClick={() => handleSort('file_size')}>
                     <div className="flex items-center gap-1">Size <ArrowUpDown className="h-3 w-3" /></div>
                  </th>
                  <th className="px-4 py-3 w-28">Status</th>
                  <th className="px-4 py-3 w-16 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border bg-card">
                {isLoading ? (
                  <tr>
                    <td colSpan={5} className="p-8 text-center text-muted-foreground">
                      <div className="flex justify-center mb-2"><Loader2 className="h-6 w-6 animate-spin opacity-50" /></div>
                      Loading documents...
                    </td>
                  </tr>
                ) : paginatedDocs.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="p-12 text-center text-muted-foreground">
                       <FileText className="h-10 w-10 mx-auto mb-3 opacity-20" />
                       <p>{searchTerm ? "No documents found matching your search" : "No documents uploaded yet"}</p>
                    </td>
                  </tr>
                ) : (
                  paginatedDocs.map((doc) => (
                    <tr key={doc.id} className="hover:bg-muted/30 transition-colors group">
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-3">
                          <div className="bg-primary/10 p-2 rounded shrink-0">
                            <FileText className="h-4 w-4 text-primary" />
                          </div>
                          <span className="font-medium truncate max-w-[200px] sm:max-w-xs" title={doc.filename}>{doc.filename}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-muted-foreground tabular-nums whitespace-nowrap">
                        {new Date(doc.created_at).toLocaleDateString()}
                      </td>
                      <td className="px-4 py-3 text-muted-foreground tabular-nums whitespace-nowrap">
                        {formatSize(doc.file_size)}
                      </td>
                      <td className="px-4 py-3">
                         <span className={cn(
                          "text-[10px] px-2 py-0.5 rounded-full uppercase tracking-wider font-semibold",
                          doc.status === 'completed' && "bg-green-500/15 text-green-600 dark:text-green-400",
                          doc.status === 'processing' && "bg-blue-500/15 text-blue-600 dark:text-blue-400",
                          doc.status === 'failed' && "bg-red-500/15 text-red-600 dark:text-red-400",
                          doc.status === 'pending' && "bg-yellow-500/15 text-yellow-600 dark:text-yellow-400",
                        )}>
                          {doc.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          className="h-8 w-8 text-muted-foreground hover:text-destructive opacity-100 sm:opacity-0 group-hover:opacity-100 transition-opacity"
                          onClick={() => handleDelete(doc.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between pt-2">
             <div className="text-sm text-muted-foreground">
               Showing {Math.min(filteredAndSortedDocs.length, (currentPage - 1) * itemsPerPage + 1)} to {Math.min(filteredAndSortedDocs.length, currentPage * itemsPerPage)} of {filteredAndSortedDocs.length} entries
             </div>
             <div className="flex items-center gap-2">
               <Button 
                 variant="outline" 
                 size="sm" 
                 onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                 disabled={currentPage === 1}
                 className="h-8 w-8 p-0"
               >
                 <ChevronLeft className="h-4 w-4" />
               </Button>
               <span className="text-sm font-medium min-w-[3rem] text-center">
                 Page {currentPage} of {totalPages}
               </span>
               <Button 
                 variant="outline" 
                 size="sm" 
                 onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                 disabled={currentPage === totalPages}
                 className="h-8 w-8 p-0"
               >
                 <ChevronRight className="h-4 w-4" />
               </Button>
             </div>
          </div>

        </div>
      </div>
    </div>
  );
}
