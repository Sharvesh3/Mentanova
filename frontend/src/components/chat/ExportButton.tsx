import { useState } from 'react';
import { Download, FileText, FileJson, Loader2 } from 'lucide-react';
import api from '../../services/api';

interface ExportButtonProps {
  conversationId: string;
}

export default function ExportButton({ conversationId }: ExportButtonProps) {
  const [loading, setLoading] = useState(false);
  const [showMenu, setShowMenu] = useState(false);

  const handleExport = async (format: 'markdown' | 'json') => {
    setLoading(true);
    setShowMenu(false);

    try {
      const data = await api.exportConversation(conversationId, format);

      if (format === 'markdown') {
        // Create download link for markdown
        const url = window.URL.createObjectURL(new Blob([data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `conversation_${conversationId}.md`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
      } else {
        // Create download link for JSON
        const blob = new Blob([JSON.stringify(data, null, 2)], {
          type: 'application/json',
        });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `conversation_${conversationId}.json`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
      }

      console.log(`✅ Exported conversation as ${format.toUpperCase()}`);
    } catch (error) {
      console.error('Export failed:', error);
      alert('Failed to export conversation. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setShowMenu(!showMenu)}
        disabled={loading}
        className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 border border-gray-300 rounded-lg transition-colors disabled:opacity-50"
        title="Export conversation"
      >
        {loading ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <Download className="w-4 h-4" />
        )}
        <span className="hidden sm:inline">Export</span>
      </button>

      {showMenu && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setShowMenu(false)}
          />

          {/* Menu */}
          <div className="absolute right-0 top-full mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 z-20">
            <div className="p-1">
              <button
                onClick={() => handleExport('markdown')}
                className="w-full flex items-center gap-3 px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded-md transition-colors"
              >
                <FileText className="w-4 h-4 text-blue-600" />
                <div className="text-left">
                  <p className="font-medium">Markdown</p>
                  <p className="text-xs text-gray-500">Formatted text</p>
                </div>
              </button>

              <button
                onClick={() => handleExport('json')}
                className="w-full flex items-center gap-3 px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded-md transition-colors"
              >
                <FileJson className="w-4 h-4 text-green-600" />
                <div className="text-left">
                  <p className="font-medium">JSON</p>
                  <p className="text-xs text-gray-500">With metadata</p>
                </div>
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}