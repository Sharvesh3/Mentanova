import { useState, useEffect } from 'react';
import { 
  X, 
  MessageSquare, 
  FileText, 
  Clock, 
  TrendingUp, 
  Calendar,
  BarChart3,
  CheckCircle,
  AlertCircle,
  MinusCircle
} from 'lucide-react';
import api from '../../services/api';

interface AnalyticsModalProps {
  conversationId: string;
  isOpen: boolean;
  onClose: () => void;
}

interface ConversationAnalytics {
  conversation_id: string;
  total_messages: number;
  user_queries: number;
  ai_responses: number;
  documents_referenced: string[];
  total_documents: number;
  total_sources_cited: number;
  confidence_distribution: {
    high: number;
    medium: number;
    low: number;
  };
  primary_document: string | null;
  active_documents: string[];
  topics: string[];
  time_periods_discussed: string[];
  created_at: string;
  duration_minutes: number;
}

export default function AnalyticsModal({ conversationId, isOpen, onClose }: AnalyticsModalProps) {
  const [analytics, setAnalytics] = useState<ConversationAnalytics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && conversationId) {
      fetchAnalytics();
    }
  }, [isOpen, conversationId]);

   const fetchAnalytics = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await api.getConversationAnalytics(conversationId);
      setAnalytics(data);
      console.log('📊 Analytics loaded:', data);
    } catch (err: any) {
      console.error('❌ Failed to fetch analytics:', err);
      setError(err.response?.data?.detail || 'Failed to load analytics');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  const getConfidencePercentage = (count: number, total: number) => {
    return total > 0 ? Math.round((count / total) * 100) : 0;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black bg-opacity-50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-purple-50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">Conversation Analytics</h2>
              <p className="text-sm text-gray-600">Insights and statistics</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Content */}
        <div className="overflow-y-auto max-h-[calc(90vh-80px)]">
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <div className="text-center">
                <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-gray-600">Loading analytics...</p>
              </div>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center py-20">
              <div className="text-center">
                <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                <p className="text-red-600 font-medium">{error}</p>
                <button
                  onClick={fetchAnalytics}
                  className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  Retry
                </button>
              </div>
            </div>
          ) : analytics ? (
            <div className="p-6 space-y-6">
              {/* Overview Stats */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {/* Total Messages */}
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-4 border border-blue-200">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center">
                      <MessageSquare className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <p className="text-sm text-blue-700 font-medium">Messages</p>
                      <p className="text-2xl font-bold text-blue-900">{analytics.total_messages}</p>
                    </div>
                  </div>
                </div>

                {/* Documents */}
                <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-4 border border-green-200">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-green-600 rounded-lg flex items-center justify-center">
                      <FileText className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <p className="text-sm text-green-700 font-medium">Documents</p>
                      <p className="text-2xl font-bold text-green-900">{analytics.total_documents}</p>
                    </div>
                  </div>
                </div>

                {/* Duration */}
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-4 border border-purple-200">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-purple-600 rounded-lg flex items-center justify-center">
                      <Clock className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <p className="text-sm text-purple-700 font-medium">Duration</p>
                      <p className="text-2xl font-bold text-purple-900">{analytics.duration_minutes}m</p>
                    </div>
                  </div>
                </div>

                {/* Sources */}
                <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-xl p-4 border border-orange-200">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-orange-600 rounded-lg flex items-center justify-center">
                      <TrendingUp className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <p className="text-sm text-orange-700 font-medium">Sources</p>
                      <p className="text-2xl font-bold text-orange-900">{analytics.total_sources_cited}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Confidence Distribution */}
              <div className="bg-white rounded-xl border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-blue-600" />
                  Response Confidence
                </h3>
                
                <div className="space-y-3">
                  {/* High Confidence */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-600" />
                        <span className="text-sm font-medium text-gray-700">High Confidence</span>
                      </div>
                      <span className="text-sm font-bold text-gray-900">
                        {analytics.confidence_distribution.high} ({getConfidencePercentage(analytics.confidence_distribution.high, analytics.ai_responses)}%)
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${getConfidencePercentage(analytics.confidence_distribution.high, analytics.ai_responses)}%` }}
                      />
                    </div>
                  </div>

                  {/* Medium Confidence */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <MinusCircle className="w-4 h-4 text-yellow-600" />
                        <span className="text-sm font-medium text-gray-700">Medium Confidence</span>
                      </div>
                      <span className="text-sm font-bold text-gray-900">
                        {analytics.confidence_distribution.medium} ({getConfidencePercentage(analytics.confidence_distribution.medium, analytics.ai_responses)}%)
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-yellow-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${getConfidencePercentage(analytics.confidence_distribution.medium, analytics.ai_responses)}%` }}
                      />
                    </div>
                  </div>

                  {/* Low Confidence */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <AlertCircle className="w-4 h-4 text-red-600" />
                        <span className="text-sm font-medium text-gray-700">Low Confidence</span>
                      </div>
                      <span className="text-sm font-bold text-gray-900">
                        {analytics.confidence_distribution.low} ({getConfidencePercentage(analytics.confidence_distribution.low, analytics.ai_responses)}%)
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-red-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${getConfidencePercentage(analytics.confidence_distribution.low, analytics.ai_responses)}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Documents Referenced */}
              {analytics.documents_referenced.length > 0 && (
                <div className="bg-white rounded-xl border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <FileText className="w-5 h-5 text-blue-600" />
                    Documents Referenced
                  </h3>
                  <div className="space-y-2">
                    {analytics.documents_referenced.map((doc, idx) => (
                      <div 
                        key={idx}
                        className={`flex items-center gap-3 px-4 py-3 rounded-lg ${
                          doc === analytics.primary_document 
                            ? 'bg-blue-50 border border-blue-200' 
                            : 'bg-gray-50 border border-gray-200'
                        }`}
                      >
                        <FileText className={`w-4 h-4 ${
                          doc === analytics.primary_document ? 'text-blue-600' : 'text-gray-400'
                        }`} />
                        <span className="flex-1 text-sm font-medium text-gray-700">{doc}</span>
                        {doc === analytics.primary_document && (
                          <span className="px-2 py-1 bg-blue-600 text-white text-xs font-medium rounded">
                            Primary
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Time Periods */}
              {analytics.time_periods_discussed.length > 0 && (
                <div className="bg-white rounded-xl border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <Calendar className="w-5 h-5 text-blue-600" />
                    Time Periods Discussed
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {analytics.time_periods_discussed.map((period, idx) => (
                      <span 
                        key={idx}
                        className="px-3 py-1.5 bg-purple-100 text-purple-700 rounded-lg text-sm font-medium"
                      >
                        {period}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Conversation Details */}
              <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-600">User Queries</p>
                    <p className="font-semibold text-gray-900">{analytics.user_queries}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">AI Responses</p>
                    <p className="font-semibold text-gray-900">{analytics.ai_responses}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Created</p>
                    <p className="font-semibold text-gray-900">
                      {new Date(analytics.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600">Conversation ID</p>
                    <p className="font-mono text-xs text-gray-600 truncate">
                      {analytics.conversation_id}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : null}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 px-6 py-4 bg-gray-50">
          <div className="flex justify-end gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}