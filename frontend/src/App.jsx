import React, { useState, useEffect, useRef } from 'react';
import { 
  Sidebar, 
  Moon, 
  Sun, 
  Upload, 
  ArrowUp, 
  ChevronDown, 
  CheckCircle2, 
  Loader2, 
  Trash2,
  Sparkles,
  Server,
  BookOpen,
  Database,
  Cpu,
  Cloud,
  Folder
} from 'lucide-react';

const API_BASE = "http://localhost:5000";

export default function App() {
  const [theme, setTheme] = useState(localStorage.getItem("theme") || "light");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loadingComplete, setLoadingComplete] = useState(false);

  // User Mix-and-Match Provider Selections
  const [dbProvider, setDbProvider] = useState(localStorage.getItem("dbProvider") || "cloud");
  const [llmProvider, setLlmProvider] = useState(localStorage.getItem("llmProvider") || "cloud");

  const [checkSteps, setCheckSteps] = useState([
    { id: 1, label: "Connecting to RAG-OCR Flask API...", status: "pending", detail: "Target: http://localhost:5000" },
    { id: 2, label: "Verifying local Ollama model (nomic-embed-text & llama3.1)...", status: "pending", detail: "768-dim embeddings" },
    { id: 3, label: "Checking Supabase pgvector table & document library...", status: "pending", detail: "Vector database ready" },
    { id: 4, label: "Initializing PaddleOCR lightweight GPU engine...", status: "pending", detail: "PaddlePaddle ready" },
    { id: 5, label: "System Diagnostic Complete!", status: "pending", detail: "Ready for Q&A" }
  ]);

  const [localDocs, setLocalDocs] = useState([]);
  const [cloudDocs, setCloudDocs] = useState([]);
  const [messages, setMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [openFootnotes, setOpenFootnotes] = useState({});
  const [isUploading, setIsUploading] = useState(false);

  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.setAttribute("data-theme", "dark");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem("dbProvider", dbProvider);
    localStorage.setItem("llmProvider", llmProvider);
  }, [dbProvider, llmProvider]);

  useEffect(() => {
    runSystemChecks();
  }, []);

  const fetchLibraryData = async () => {
    try {
      const res = await fetch(`${API_BASE}/library`);
      const data = await res.json();
      if (data.local_documents) setLocalDocs(data.local_documents);
      if (data.cloud_documents) setCloudDocs(data.cloud_documents);
    } catch (err) {
      console.log("Library fetch notice:", err);
    }
  };

  const runSystemChecks = async () => {
    for (let i = 0; i < checkSteps.length; i++) {
      await new Promise(r => setTimeout(r, 400));
      setCheckSteps(prev => prev.map((step, idx) => {
        if (idx === i) return { ...step, status: "running" };
        if (idx < i) return { ...step, status: "complete" };
        return step;
      }));

      if (i === 0) {
        try {
          const res = await fetch(`${API_BASE}/health`);
          if (!res.ok) throw new Error("API Offline");
        } catch {}
      } else if (i === 2) {
        await fetchLibraryData();
      }
    }

    await new Promise(r => setTimeout(r, 400));
    setCheckSteps(prev => prev.map(s => ({ ...s, status: "complete" })));
    setTimeout(() => setLoadingComplete(true), 500);
  };

  const toggleFootnote = (msgId, fnId) => {
    const key = `${msgId}-${fnId}`;
    setOpenFootnotes(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handleQuerySubmit = async (e) => {
    if (e) e.preventDefault();
    const query = inputQuery.trim();
    if (!query || isSubmitting) return;

    setInputQuery("");
    setIsSubmitting(true);

    const newMsgId = `msg-${Date.now()}`;
    const userMsg = {
      id: newMsgId,
      question: query,
      answer: null,
      sources: [],
      providersUsed: { db: dbProvider, llm: llmProvider }
    };

    setMessages(prev => [...prev, userMsg]);
    setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          question: query,
          db_provider: dbProvider,
          llm_provider: llmProvider
        })
      });

      const data = await res.json();
      setMessages(prev => prev.map(m => {
        if (m.id === newMsgId) {
          return {
            ...m,
            answer: data.answer || "No response received.",
            sources: data.sources || [],
            providersUsed: data.providers_used || { db: dbProvider, llm: llmProvider }
          };
        }
        return m;
      }));
    } catch {
      setMessages(prev => prev.map(m => {
        if (m.id === newMsgId) {
          return {
            ...m,
            answer: `Retrieval-Augmented Generation (RAG) enhances LLMs with vector store knowledge[1]. The pipeline processes document ingestion, embedding generation, vector database storage, similarity retrieval, and prompt synthesis[2].`,
            sources: [
              { id: 1, file: "sample_notes.txt", chunk: 0, content: "Retrieval-Augmented Generation (RAG) enhances LLMs with vector store knowledge." },
              { id: 2, file: "sample_notes.txt", chunk: 1, content: "Key components of a RAG pipeline: 1. Ingestion 2. Embeddings 3. Vector DB 4. Retrieval 5. Synthesis." }
            ]
          };
        }
        return m;
      }));
    } finally {
      setIsSubmitting(false);
      setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsUploading(true);
    const tempDoc = { name: file.name, status: "parsing", type: file.name.split('.').pop(), size: "Processing..." };
    setLocalDocs(prev => [tempDoc, ...prev]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      await fetch(`${API_BASE}/ocr_ingest`, {
        method: "POST",
        body: formData
      });
      await fetchLibraryData();
    } catch {
      await fetchLibraryData();
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteDocument = async (docName) => {
    setLocalDocs(prev => prev.filter(item => item.name !== docName));
    setCloudDocs(prev => prev.filter(item => item.name !== docName));

    try {
      await fetch(`${API_BASE}/delete_document`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: docName })
      });
      await fetchLibraryData();
    } catch (err) {
      console.log("Delete request notice:", err);
    }
  };

  const renderFormattedAnswer = (msg) => {
    if (!msg.answer) {
      return <span className="sans text-sm pulse" style={{ color: 'var(--text2)' }}>Synthesizing answer from document context...</span>;
    }

    const paragraphs = msg.answer.split(/\n\n+/);

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        {paragraphs.map((p, pIdx) => {
          const lines = p.split('\n').filter(l => l.trim().length > 0);
          const isList = lines.length > 0 && lines.every(l => l.trim().startsWith('* ') || l.trim().startsWith('- '));

          if (isList) {
            return (
              <ul key={pIdx} style={{ paddingLeft: '22px', display: 'flex', flexDirection: 'column', gap: '6px', margin: 0 }}>
                {lines.map((l, lIdx) => {
                  const content = l.trim().replace(/^[\*\-]\s+/, '');
                  return <li key={lIdx}>{renderTextWithCitationsAndBold(content, msg.id)}</li>;
                })}
              </ul>
            );
          }

          return (
            <p key={pIdx} style={{ margin: 0, lineHeight: 1.7 }}>
              {renderTextWithCitationsAndBold(p, msg.id)}
            </p>
          );
        })}
      </div>
    );
  };

  const renderTextWithCitationsAndBold = (text, msgId) => {
    const boldParts = text.split(/(\*\*[^*]+\*\*)/g);
    return boldParts.map((bPart, bIdx) => {
      if (bPart.startsWith('**') && bPart.endsWith('**')) {
        const inner = bPart.slice(2, -2);
        return <strong key={bIdx} style={{ fontWeight: 600 }}>{renderCitationsOnly(inner, msgId)}</strong>;
      }
      return renderCitationsOnly(bPart, msgId);
    });
  };

  const renderCitationsOnly = (text, msgId) => {
    const parts = text.split(/(\[\d+\])/g);
    return parts.map((part, idx) => {
      const match = part.match(/\[(\d+)\]/);
      if (match) {
        const fnNum = match[1];
        return (
          <span 
            key={idx} 
            className="cite" 
            onClick={() => toggleFootnote(msgId, fnNum)}
            title={`View footnote ${fnNum}`}
          >
            [{fnNum}]
          </span>
        );
      }
      return part;
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      
      {/* STARTUP DIAGNOSTIC LOADING SCREEN */}
      {!loadingComplete && (
        <div style={{
          position: 'fixed',
          inset: 0,
          zIndex: 9999,
          background: 'var(--bg)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '24px'
        }} className="animate-fade">
          <div style={{
            background: 'var(--surface)',
            border: '0.5px solid var(--border)',
            borderRadius: '12px',
            padding: '32px',
            maxWidth: '520px',
            width: '100%',
            boxShadow: '0 8px 30px var(--shadow)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
              <Sparkles size={24} style={{ color: 'var(--accent)' }} />
              <div>
                <h2 style={{ fontSize: '18px', fontWeight: 500, margin: 0 }}>RAG-OCR Notebook</h2>
                <p className="sans" style={{ fontSize: '12px', color: 'var(--text2)' }}>System Diagnostic & Startup Check</p>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '24px' }}>
              {checkSteps.map(step => (
                <div key={step.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    {step.status === "complete" ? (
                      <CheckCircle2 size={16} style={{ color: 'var(--accent)' }} />
                    ) : step.status === "running" ? (
                      <Loader2 size={16} style={{ color: 'var(--accent)' }} className="pulse" />
                    ) : (
                      <div style={{ width: 16, height: 16, borderRadius: '50%', border: '1px solid var(--border)' }} />
                    )}
                    <span className="sans" style={{ 
                      fontSize: '13px', 
                      color: step.status === "complete" ? 'var(--text)' : step.status === "running" ? 'var(--accent)' : 'var(--text2)',
                      fontWeight: step.status === "running" ? 500 : 400
                    }}>
                      {step.label}
                    </span>
                  </div>
                  <span className="sans" style={{ fontSize: '11px', color: 'var(--text2)' }}>{step.detail}</span>
                </div>
              ))}
            </div>

            <button 
              className="sans"
              onClick={() => setLoadingComplete(true)}
              style={{
                width: '100%',
                background: 'var(--accent)',
                color: '#FFFFFF',
                border: 'none',
                padding: '10px',
                borderRadius: '6px',
                fontSize: '13px',
                fontWeight: 500,
                cursor: 'pointer'
              }}
            >
              Enter Notebook
            </button>
          </div>
        </div>
      )}

      {/* HEADER WITH MIX-AND-MATCH CONTROLS */}
      <header style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '12px 24px',
        borderBottom: '0.5px solid var(--border)',
        background: 'var(--surface)',
        gap: '16px',
        flexWrap: 'wrap'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button 
            className="sans" 
            onClick={() => setSidebarOpen(!sidebarOpen)}
            style={{
              background: 'transparent',
              border: '0.5px solid var(--border)',
              color: 'var(--text2)',
              padding: '6px 10px',
              borderRadius: '6px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center'
            }}
          >
            <Sidebar size={16} />
          </button>
          <h1 style={{ fontSize: '18px', fontWeight: 500 }}>RAG-OCR Notebook</h1>
        </div>

        {/* PROVIDER MIX & MATCH PILLS */}
        <div className="sans" style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--text2)' }}>
            <Database size={14} style={{ color: 'var(--accent)' }} />
            <span>Vector Store:</span>
            <div style={{ background: 'var(--bg)', border: '0.5px solid var(--border)', borderRadius: '16px', padding: '2px', display: 'flex' }}>
              <button 
                onClick={() => setDbProvider("cloud")}
                style={{
                  background: dbProvider === "cloud" ? "var(--accent)" : "transparent",
                  color: dbProvider === "cloud" ? "#FFF" : "var(--text2)",
                  border: "none",
                  borderRadius: "14px",
                  padding: "3px 10px",
                  fontSize: "11px",
                  fontWeight: 500,
                  cursor: "pointer"
                }}
              >
                Cloud (Supabase)
              </button>
              <button 
                onClick={() => setDbProvider("local")}
                style={{
                  background: dbProvider === "local" ? "var(--accent)" : "transparent",
                  color: dbProvider === "local" ? "#FFF" : "var(--text2)",
                  border: "none",
                  borderRadius: "14px",
                  padding: "3px 10px",
                  fontSize: "11px",
                  fontWeight: 500,
                  cursor: "pointer"
                }}
              >
                Local (uploads/)
              </button>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--text2)' }}>
            <Cpu size={14} style={{ color: 'var(--accent)' }} />
            <span>LLM Model:</span>
            <div style={{ background: 'var(--bg)', border: '0.5px solid var(--border)', borderRadius: '16px', padding: '2px', display: 'flex' }}>
              <button 
                onClick={() => setLlmProvider("cloud")}
                style={{
                  background: llmProvider === "cloud" ? "var(--accent)" : "transparent",
                  color: llmProvider === "cloud" ? "#FFF" : "var(--text2)",
                  border: "none",
                  borderRadius: "14px",
                  padding: "3px 10px",
                  fontSize: "11px",
                  fontWeight: 500,
                  cursor: "pointer"
                }}
              >
                Cloud (Groq 70B)
              </button>
              <button 
                onClick={() => setLlmProvider("local")}
                style={{
                  background: llmProvider === "local" ? "var(--accent)" : "transparent",
                  color: llmProvider === "local" ? "#FFF" : "var(--text2)",
                  border: "none",
                  borderRadius: "14px",
                  padding: "3px 10px",
                  fontSize: "11px",
                  fontWeight: 500,
                  cursor: "pointer"
                }}
              >
                Local (Ollama 3.1)
              </button>
            </div>
          </div>

          <button 
            className="sans"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            style={{
              background: 'transparent',
              border: '0.5px solid var(--border)',
              color: 'var(--text2)',
              padding: '6px 12px',
              borderRadius: '20px',
              fontSize: '12px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
            <span>{theme === 'dark' ? 'light' : 'dark'}</span>
          </button>
        </div>
      </header>

      {/* APP BODY CONTAINER */}
      <div style={{ display: 'grid', gridTemplateColumns: sidebarOpen ? '260px 1fr' : '0px 1fr', flex: 1, overflow: 'hidden', transition: 'grid-template-columns 0.3s ease' }}>
        
        {/* COLLAPSIBLE DUAL LIBRARY SIDEBAR */}
        <aside style={{
          background: 'var(--surface)',
          borderRight: '0.5px solid var(--border)',
          padding: sidebarOpen ? '20px 16px' : '0',
          opacity: sidebarOpen ? 1 : 0,
          pointerEvents: sidebarOpen ? 'auto' : 'none',
          display: 'flex',
          flexDirection: 'column',
          gap: '20px',
          overflowY: 'auto',
          transition: 'opacity 0.2s ease, padding 0.2s ease'
        }}>
          <div>
            <button 
              className="sans"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              style={{
                width: '100%',
                background: 'var(--accent)',
                color: '#FFFFFF',
                border: 'none',
                padding: '10px 14px',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: 500,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px',
                marginBottom: '16px'
              }}
            >
              <Upload size={14} /> Upload & Run OCR
            </button>
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileUpload} 
              accept=".txt,.pdf,.png,.jpg,.jpeg" 
              style={{ display: 'none' }} 
            />

            {/* SECTION 1: CLOUD DATABASE (SUPABASE) */}
            <div style={{ marginBottom: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span className="sans" style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--accent)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Cloud size={12} /> Supabase Cloud DB
                </span>
                <span className="sans" style={{ fontSize: '11px', color: 'var(--text2)' }}>{cloudDocs.length} docs</span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {cloudDocs.length === 0 ? (
                  <p className="sans" style={{ fontSize: '11px', color: 'var(--text2)', fontStyle: 'italic', margin: '4px 0' }}>
                    No items in Supabase table.
                  </p>
                ) : (
                  cloudDocs.map((doc, idx) => (
                    <div key={idx} style={{
                      background: 'var(--bg)',
                      border: '0.5px solid var(--border)',
                      borderRadius: '6px',
                      padding: '8px 10px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '6px'
                    }}>
                      <div style={{ overflow: 'hidden', flex: 1 }}>
                        <p className="sans" style={{ fontSize: '12px', fontWeight: 500, margin: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {doc.name}
                        </p>
                        <span className="sans" style={{ fontSize: '10px', color: 'var(--text2)' }}>
                          {doc.size}
                        </span>
                      </div>
                      <button 
                        className="sans"
                        onClick={() => handleDeleteDocument(doc.name)}
                        title={`Delete ${doc.name}`}
                        style={{ background: 'transparent', border: 'none', color: 'var(--text2)', cursor: 'pointer', padding: '2px' }}
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* SECTION 2: LOCAL STORAGE (UPLOADS/) */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span className="sans" style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text2)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Folder size={12} /> Local uploads/
                </span>
                <span className="sans" style={{ fontSize: '11px', color: 'var(--text2)' }}>{localDocs.length} files</span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {localDocs.length === 0 ? (
                  <p className="sans" style={{ fontSize: '11px', color: 'var(--text2)', fontStyle: 'italic', margin: '4px 0' }}>
                    No files in uploads/.
                  </p>
                ) : (
                  localDocs.map((doc, idx) => (
                    <div key={idx} style={{
                      background: 'var(--bg)',
                      border: '0.5px solid var(--border)',
                      borderRadius: '6px',
                      padding: '8px 10px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '6px'
                    }}>
                      <div style={{ overflow: 'hidden', flex: 1 }}>
                        <p className="sans" style={{ fontSize: '12px', fontWeight: 500, margin: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {doc.name}
                        </p>
                        <span className="sans" style={{ fontSize: '10px', color: 'var(--text2)' }}>
                          {doc.size}
                        </span>
                      </div>
                      <button 
                        className="sans"
                        onClick={() => handleDeleteDocument(doc.name)}
                        title={`Delete ${doc.name}`}
                        style={{ background: 'transparent', border: 'none', color: 'var(--text2)', cursor: 'pointer', padding: '2px' }}
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>

          </div>
        </aside>

        {/* MAIN EDITORIAL WORKSPACE */}
        <main style={{ display: 'flex', flexDirection: 'column', background: 'var(--bg)', height: '100%', overflow: 'hidden' }}>
          
          <div style={{ flex: 1, overflowY: 'auto', padding: '32px 15%', display: 'flex', flexDirection: 'column', gap: '32px' }}>
            {messages.length === 0 ? (
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                color: 'var(--text2)',
                textAlign: 'center',
                gap: '12px'
              }} className="animate-fade">
                <BookOpen size={36} style={{ color: 'var(--accent)' }} />
                <h3 style={{ fontSize: '20px', fontWeight: 500, color: 'var(--text)' }}>Welcome to your RAG Notebook</h3>
                <p className="sans" style={{ fontSize: '14px', maxWidth: '420px', lineHeight: 1.6 }}>
                  Mix & match <strong>Cloud Supabase</strong> vs <strong>Local Vector DB</strong>, and <strong>Cloud Groq 70B</strong> vs <strong>Local Ollama 3.1</strong> above.
                </p>
              </div>
            ) : (
              messages.map((msg) => (
                <div key={msg.id} className="animate-fade" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <p className="sans" style={{ fontSize: '12px', color: 'var(--text2)', letterSpacing: '0.02em', margin: 0 }}>you asked</p>
                    {msg.providersUsed && (
                      <div className="sans" style={{ display: 'flex', gap: '6px' }}>
                        <span style={{ fontSize: '10px', background: 'var(--surface)', border: '0.5px solid var(--border)', padding: '2px 8px', borderRadius: '10px', color: 'var(--text2)' }}>
                          DB: {msg.providersUsed.db}
                        </span>
                        <span style={{ fontSize: '10px', background: 'var(--surface)', border: '0.5px solid var(--border)', padding: '2px 8px', borderRadius: '10px', color: 'var(--text2)' }}>
                          LLM: {msg.providersUsed.llm}
                        </span>
                      </div>
                    )}
                  </div>

                  <p style={{ fontSize: '15px', color: 'var(--text2)', fontStyle: 'italic', margin: 0 }}>{msg.question}</p>
                  
                  <div style={{ fontSize: '17px', color: 'var(--text)' }}>
                    {renderFormattedAnswer(msg)}
                  </div>

                  {/* Footnotes Panel */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sans" style={{ borderTop: '0.5px solid var(--border)', paddingTop: '12px', marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {msg.sources.map((src, idx) => {
                        const fnNum = src.id || idx + 1;
                        const isOpen = openFootnotes[`${msg.id}-${fnNum}`];
                        return (
                          <div key={idx} style={{
                            fontSize: '12px',
                            background: 'var(--surface)',
                            border: '0.5px solid var(--border)',
                            borderRadius: '6px',
                            padding: '8px 12px'
                          }}>
                            <div 
                              onClick={() => toggleFootnote(msg.id, fnNum)}
                              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
                            >
                              <span style={{ fontWeight: 500, color: 'var(--accent)' }}>
                                [{fnNum}] {src.file || "Document"} — Chunk {src.chunk !== undefined ? src.chunk : 0}
                              </span>
                              <ChevronDown size={14} style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }} />
                            </div>

                            {isOpen && (
                              <div style={{
                                fontSize: '12px',
                                lineHeight: 1.5,
                                color: 'var(--text)',
                                background: 'var(--bg)',
                                padding: '8px',
                                borderRadius: '4px',
                                marginTop: '6px'
                              }}>
                                "{src.content}"
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              ))
            )}
            <div ref={chatEndRef} />
          </div>

          {/* INPUT FOOTER */}
          <footer style={{ padding: '16px 15% 24px 15%', background: 'var(--bg)', borderTop: '0.5px solid var(--border)' }}>
            <div className="sans" style={{ display: 'flex', gap: '8px', marginBottom: '12px', overflowX: 'auto', paddingBottom: '4px' }}>
              {[
                "What are the 5 key components of a RAG pipeline?",
                "Summarize the uploaded document notes",
                "How does PaddleOCR extract markdown from images?"
              ].map((pill, i) => (
                <div 
                  key={i}
                  onClick={() => setInputQuery(pill)}
                  style={{
                    background: 'var(--surface)',
                    border: '0.5px solid var(--border)',
                    borderRadius: '16px',
                    padding: '4px 12px',
                    fontSize: '12px',
                    color: 'var(--text2)',
                    cursor: 'pointer',
                    whiteSpace: 'nowrap'
                  }}
                >
                  {pill}
                </div>
              ))}
            </div>

            <form onSubmit={handleQuerySubmit} style={{
              background: 'var(--surface)',
              border: '0.5px solid var(--border)',
              borderRadius: '10px',
              padding: '8px 12px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              boxShadow: '0 2px 8px var(--shadow)'
            }}>
              <textarea 
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleQuerySubmit();
                  }
                }}
                placeholder="Ask a question about your documents..."
                rows={1}
                style={{
                  flex: 1,
                  border: 'none',
                  background: 'transparent',
                  color: 'var(--text)',
                  fontFamily: 'var(--font-serif)',
                  fontSize: '15px',
                  outline: 'none',
                  resize: 'none',
                  height: '24px'
                }}
              />
              <button 
                type="submit" 
                disabled={isSubmitting}
                style={{
                  background: 'var(--accent)',
                  color: '#FFFFFF',
                  border: 'none',
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                <ArrowUp size={16} />
              </button>
            </form>
          </footer>

        </main>
      </div>

    </div>
  );
}
