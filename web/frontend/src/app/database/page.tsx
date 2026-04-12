"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import API_BASE_URL from "@/config";

interface MaterialPrediction {
    material_id: string;
    formula: string;
    pred_ehull: number;
    p_stable: number;
    uncertainty: string;
    action: string;
}

interface ScreeningExecutionRow {
    rank: string;
    jarvis_id: string;
    formula: string;
    p_stable: string;
    qe_final_state_est: string;
    decision: string;
    decision_reason: string;
    action?: string;
}

interface ProofArtifact {
    id: string;
    relative_path: string;
    exists: boolean;
    size_bytes: number | null;
    mtime_utc: string | null;
    download_url: string | null;
}

interface GroundedWinMetrics {
    [key: string]: string;
}

interface ScreeningProvisionalPayload {
    success: boolean;
    mode: string;
    summary_title: string;
    summary_fields: Record<string, string>;
    counts: {
        total_candidates: number;
        accept: number;
        hold: number;
        unknown: number;
        top20_unresolved: number;
    };
    screening: {
        screen_now: ScreeningExecutionRow[];
        resolve_qe_first: ScreeningExecutionRow[];
        compact: ScreeningExecutionRow[];
    };
    grounded_win: {
        h100_ehull_ens_v1: GroundedWinMetrics;
        oqmd_ens_v1: GroundedWinMetrics;
        jarvis_ens_v1: GroundedWinMetrics;
    };
    proofs: ProofArtifact[];
}

const PROOF_LABELS: Record<string, string> = {
    screening_decision_summary: "Decision Summary",
    screening_decision_provisional: "Decision Table (Provisional)",
    screening_execution_compact: "Execution List (Compact)",
    screening_execution_accept: "Execution List (Screen Now)",
    screening_execution_must_resolve_top20: "Execution List (Resolve QE First)",
    qe_final_status_estimated: "QE Final Status (Estimated)",
    qe_ranked_final_estimated: "QE Ranked Candidates (Estimated)",
    grounded_win_h100_ehull_ens_v1: "Grounded Win (H100 e_hull ensemble)",
    grounded_win_oqmd_ens_v1: "Grounded Win (OQMD ensemble)",
    grounded_win_jarvis_ens_v1: "Grounded Win (JARVIS ensemble)",
};

function formatProbability(value: string): string {
    const p = Number.parseFloat(value);
    if (!Number.isFinite(p)) {
        return "-";
    }
    return `${(p * 100).toFixed(1)}%`;
}

function formatBytes(size: number | null): string {
    if (size === null) {
        return "-";
    }
    if (size < 1024) {
        return `${size} B`;
    }
    if (size < 1024 * 1024) {
        return `${(size / 1024).toFixed(1)} KB`;
    }
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Database() {
    const [search, setSearch] = useState("");
    const [data, setData] = useState<MaterialPrediction[]>([]);
    const [filtered, setFiltered] = useState<MaterialPrediction[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [page, setPage] = useState(1);
    const perPage = 50;

    const [screeningLoading, setScreeningLoading] = useState(true);
    const [screeningError, setScreeningError] = useState<string | null>(null);
    const [screening, setScreening] = useState<ScreeningProvisionalPayload | null>(null);
    const [proofError, setProofError] = useState<string | null>(null);

    useEffect(() => {
        fetch("/api/backend/database")
            .then((res) => res.json())
            .then((result) => {
                if (result.success) {
                    setData(result.data);
                    setFiltered(result.data);
                } else {
                    setError(result.error || "Failed to load data");
                }
                setLoading(false);
            })
            .catch(() => {
                setError("Failed to connect to server");
                setLoading(false);
            });
    }, []);

    useEffect(() => {
        fetch("/api/backend/screening/provisional")
            .then((res) => {
                if (!res.ok) {
                    throw new Error(`Request failed (${res.status})`);
                }
                return res.json();
            })
            .then((result: ScreeningProvisionalPayload) => {
                setScreening(result);
                setScreeningError(null);
                setScreeningLoading(false);
            })
            .catch((err: Error) => {
                setScreeningError(err.message || "Failed to load provisional screening pack");
                setScreeningLoading(false);
            });
    }, []);

    useEffect(() => {
        const q = search.toLowerCase();
        const results = data.filter(
            (m) =>
                m.material_id.toLowerCase().includes(q) ||
                m.formula.toLowerCase().includes(q)
        );
        setFiltered(results);
        setPage(1);
    }, [search, data]);

    const stats = {
        total: data.length,
        dft: data.filter((m) => m.action === "DFT").length,
        hold: data.filter((m) => m.action === "HOLD").length,
        skip: data.filter((m) => m.action === "SKIP").length,
    };

    const downloadCSV = () => {
        const csvHeaders = ["material_id", "formula", "pred_ehull", "p_stable", "uncertainty", "action"];
        const rows = filtered.map((m) =>
            [m.material_id, m.formula, m.pred_ehull, m.p_stable, m.uncertainty, m.action].join(",")
        );
        const csv = [csvHeaders.join(","), ...rows].join("\n");
        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "cathodescreen_predictions.csv";
        a.click();
    };

    const downloadProof = async (proof: ProofArtifact) => {
        if (!proof.download_url) {
            return;
        }
        setProofError(null);
        try {
            const target = proof.download_url.startsWith("http")
                ? proof.download_url
                : `/api/backend${proof.download_url}`;
            const res = await fetch(target);
            if (!res.ok) {
                throw new Error(`Download failed (${res.status})`);
            }
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = proof.relative_path.split("/").pop() || proof.id;
            link.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            setProofError(err instanceof Error ? err.message : "Failed to download proof artifact");
        }
    };

    const totalPages = Math.ceil(filtered.length / perPage);
    const displayed = filtered.slice((page - 1) * perPage, page * perPage);

    return (
        <div className="min-h-screen bg-white">
            <header className="fixed w-full top-4 z-50 flex justify-center pointer-events-none">
                <div className="bg-white/80 backdrop-blur-md border border-white/50 shadow-lg rounded-full px-8 py-3 flex items-center gap-12 pointer-events-auto">
                    <Link href="/" className="font-bold text-gray-900 text-lg tracking-tight hover:text-blue-600 hover:scale-125 transition-all duration-200">
                        CathodeScreen
                    </Link>
                    <nav className="flex items-center gap-8 text-sm font-medium text-gray-600">
                        <Link href="/predict" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">Predict</Link>
                        <Link href="/database" className="text-blue-600 hover:scale-125 transition-all duration-200">Database</Link>
                        <Link href="/database#screening-pack" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">Screening</Link>
                        <Link href="/discovery" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">Discovery</Link>
                        <Link href="/about" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">About</Link>
                    </nav>
                    <div className="pl-8 border-l border-gray-200">
                        <a href={`${API_BASE_URL}/docs`} target="_blank" className="text-xs font-bold text-blue-600 hover:text-blue-700 hover:scale-125 transition-all duration-200 uppercase tracking-wide flex items-center gap-1">
                            API <span className="text-[10px]">-&gt;</span>
                        </a>
                    </div>
                </div>
            </header>

            <main className="max-w-6xl mx-auto px-6 py-10 pt-32">
                <div className="flex justify-between items-start mb-6">
                    <div>
                        <h1 className="text-2xl font-semibold text-gray-900 mb-2">
                            Pre-computed Predictions
                        </h1>
                        <p className="text-gray-600">
                            Browse stability predictions for cathode materials from Materials Project.
                        </p>
                    </div>
                    <button
                        onClick={downloadCSV}
                        disabled={loading || filtered.length === 0}
                        className="px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
                    >
                        Download CSV
                    </button>
                </div>

                {loading ? (
                    <div className="text-center py-12 text-gray-500">Loading...</div>
                ) : error ? (
                    <div className="text-center py-12 text-red-500">{error}</div>
                ) : (
                    <>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                            <div className="border border-gray-200 rounded-lg p-4">
                                <p className="text-2xl font-semibold text-gray-900">{stats.total.toLocaleString()}</p>
                                <p className="text-sm text-gray-500">Total Materials</p>
                            </div>
                            <div className="border border-gray-200 rounded-lg p-4">
                                <p className="text-2xl font-semibold text-green-600">{stats.dft.toLocaleString()}</p>
                                <p className="text-sm text-gray-500">Recommended for DFT</p>
                            </div>
                            <div className="border border-gray-200 rounded-lg p-4">
                                <p className="text-2xl font-semibold text-yellow-600">{stats.hold.toLocaleString()}</p>
                                <p className="text-sm text-gray-500">Hold for Review</p>
                            </div>
                            <div className="border border-gray-200 rounded-lg p-4">
                                <p className="text-2xl font-semibold text-gray-400">{stats.skip.toLocaleString()}</p>
                                <p className="text-sm text-gray-500">Skip</p>
                            </div>
                        </div>

                        <div className="mb-4">
                            <input
                                type="text"
                                placeholder="Search by material ID or formula..."
                                value={search}
                                onChange={(e) => setSearch(e.target.value)}
                                className="w-full max-w-md px-4 py-2 border border-gray-300 rounded-lg text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            />
                        </div>

                        <div className="border border-gray-200 rounded-lg overflow-hidden">
                            <table className="w-full text-sm">
                                <thead className="bg-gray-50 border-b border-gray-200">
                                    <tr>
                                        <th className="px-4 py-3 text-left font-medium text-gray-600">Material ID</th>
                                        <th className="px-4 py-3 text-left font-medium text-gray-600">Formula</th>
                                        <th className="px-4 py-3 text-right font-medium text-gray-600">E<sub>hull</sub> (eV)</th>
                                        <th className="px-4 py-3 text-right font-medium text-gray-600">P(Stable)</th>
                                        <th className="px-4 py-3 text-center font-medium text-gray-600">Uncertainty</th>
                                        <th className="px-4 py-3 text-center font-medium text-gray-600">Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {displayed.map((m, i) => (
                                        <tr key={m.material_id} className={i % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                                            <td className="px-4 py-3 font-mono text-blue-700 font-medium">
                                                <a
                                                    href={`https://materialsproject.org/materials/${m.material_id}`}
                                                    target="_blank"
                                                    className="hover:underline"
                                                >
                                                    {m.material_id}
                                                </a>
                                            </td>
                                            <td className="px-4 py-3 text-gray-900 font-medium">{m.formula || "-"}</td>
                                            <td className="px-4 py-3 text-right font-mono text-gray-800">{m.pred_ehull.toFixed(3)}</td>
                                            <td className="px-4 py-3 text-right text-gray-800">{(m.p_stable * 100).toFixed(0)}%</td>
                                            <td className="px-4 py-3 text-center">
                                                <span className={`text-xs ${m.uncertainty === "Low" ? "text-green-600" :
                                                    m.uncertainty === "Medium" ? "text-yellow-600" : "text-red-600"
                                                    }`}>
                                                    {m.uncertainty}
                                                </span>
                                            </td>
                                            <td className="px-4 py-3 text-center">
                                                <span className={`px-2 py-1 rounded text-xs font-medium ${m.action === "DFT" ? "bg-green-100 text-green-800" :
                                                    m.action === "HOLD" ? "bg-yellow-100 text-yellow-800" :
                                                        "bg-gray-100 text-gray-600"
                                                    }`}>
                                                    {m.action}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        <div className="mt-4 flex items-center justify-between text-sm text-gray-500">
                            <p>
                                Showing {(page - 1) * perPage + 1}-{Math.min(page * perPage, filtered.length)} of {filtered.length.toLocaleString()} materials
                            </p>
                            <div className="flex gap-2">
                                <button
                                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                                    disabled={page === 1}
                                    className="px-3 py-1 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
                                >
                                    Previous
                                </button>
                                <span className="px-3 py-1">
                                    Page {page} of {totalPages}
                                </span>
                                <button
                                    onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                                    disabled={page === totalPages}
                                    className="px-3 py-1 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
                                >
                                    Next
                                </button>
                            </div>
                        </div>
                    </>
                )}

                <section id="screening-pack" className="mt-16">
                    <div className="flex items-start justify-between mb-4">
                        <div>
                            <h2 className="text-2xl font-semibold text-gray-900">Production Screening Pack</h2>
                            <p className="text-gray-600">Provisional JARVIS-50 shortlist with explicit action gates and proof artifacts.</p>
                        </div>
                    </div>

                    {screeningLoading ? (
                        <div className="border border-gray-200 rounded-lg p-6 text-gray-500">Loading provisional screening pack...</div>
                    ) : screeningError ? (
                        <div className="border border-red-200 bg-red-50 rounded-lg p-6 text-red-700">{screeningError}</div>
                    ) : screening ? (
                        <>
                            <div className="border border-amber-200 bg-amber-50 rounded-lg p-4 mb-6">
                                <p className="text-sm text-amber-900 font-medium">
                                    {screening.summary_title || "Screening Decision Summary"}
                                </p>
                                <p className="text-sm text-amber-800 mt-1">
                                    {screening.summary_fields.note || "Provisional status; unresolved top-20 entries must finish QE before final lock."}
                                </p>
                            </div>

                            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
                                <div className="border border-gray-200 rounded-lg p-4">
                                    <p className="text-2xl font-semibold text-gray-900">{screening.counts.total_candidates}</p>
                                    <p className="text-sm text-gray-500">Candidates</p>
                                </div>
                                <div className="border border-green-200 rounded-lg p-4">
                                    <p className="text-2xl font-semibold text-green-700">{screening.counts.accept}</p>
                                    <p className="text-sm text-gray-500">Screen Now</p>
                                </div>
                                <div className="border border-amber-200 rounded-lg p-4">
                                    <p className="text-2xl font-semibold text-amber-700">{screening.counts.hold}</p>
                                    <p className="text-sm text-gray-500">Hold</p>
                                </div>
                                <div className="border border-gray-200 rounded-lg p-4">
                                    <p className="text-2xl font-semibold text-gray-700">{screening.counts.unknown}</p>
                                    <p className="text-sm text-gray-500">Unknown</p>
                                </div>
                                <div className="border border-red-200 rounded-lg p-4">
                                    <p className="text-2xl font-semibold text-red-700">{screening.counts.top20_unresolved}</p>
                                    <p className="text-sm text-gray-500">Top-20 Unresolved</p>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                                <div className="border border-gray-200 rounded-lg overflow-hidden">
                                    <div className="px-4 py-3 bg-green-50 border-b border-gray-200">
                                        <p className="text-sm font-semibold text-green-800">Action: screen_now</p>
                                    </div>
                                    <div className="max-h-80 overflow-y-auto">
                                        <table className="w-full text-sm">
                                            <thead className="bg-gray-50 border-b border-gray-200 sticky top-0">
                                                <tr>
                                                    <th className="px-3 py-2 text-left font-medium text-gray-600">Rank</th>
                                                    <th className="px-3 py-2 text-left font-medium text-gray-600">JARVIS ID</th>
                                                    <th className="px-3 py-2 text-left font-medium text-gray-600">Formula</th>
                                                    <th className="px-3 py-2 text-right font-medium text-gray-600">P(Stable)</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {screening.screening.screen_now.map((row, idx) => (
                                                    <tr key={`${row.jarvis_id}-${idx}`} className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                                                        <td className="px-3 py-2 font-mono text-gray-700">{row.rank}</td>
                                                        <td className="px-3 py-2 font-mono text-blue-700">{row.jarvis_id}</td>
                                                        <td className="px-3 py-2 text-gray-900">{row.formula || "-"}</td>
                                                        <td className="px-3 py-2 text-right text-gray-800">{formatProbability(row.p_stable)}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>

                                <div className="border border-gray-200 rounded-lg overflow-hidden">
                                    <div className="px-4 py-3 bg-amber-50 border-b border-gray-200">
                                        <p className="text-sm font-semibold text-amber-800">Action: resolve_qe_first</p>
                                    </div>
                                    <div className="max-h-80 overflow-y-auto">
                                        <table className="w-full text-sm">
                                            <thead className="bg-gray-50 border-b border-gray-200 sticky top-0">
                                                <tr>
                                                    <th className="px-3 py-2 text-left font-medium text-gray-600">Rank</th>
                                                    <th className="px-3 py-2 text-left font-medium text-gray-600">JARVIS ID</th>
                                                    <th className="px-3 py-2 text-left font-medium text-gray-600">QE State</th>
                                                    <th className="px-3 py-2 text-left font-medium text-gray-600">Reason</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {screening.screening.resolve_qe_first.map((row, idx) => (
                                                    <tr key={`${row.jarvis_id}-${idx}`} className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                                                        <td className="px-3 py-2 font-mono text-gray-700">{row.rank}</td>
                                                        <td className="px-3 py-2 font-mono text-blue-700">{row.jarvis_id}</td>
                                                        <td className="px-3 py-2 text-gray-800">{row.qe_final_state_est || "-"}</td>
                                                        <td className="px-3 py-2 text-gray-800">{row.decision_reason || "-"}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>

                            <div className="border border-gray-200 rounded-lg p-4 mb-8">
                                <h3 className="text-lg font-semibold text-gray-900 mb-4">Proof Metrics</h3>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                                    <div className="border border-gray-200 rounded-lg p-4">
                                        <p className="font-medium text-gray-900 mb-2">H100 e_hull Ensemble</p>
                                        <p className="text-gray-700">EF@1%: {screening.grounded_win.h100_ehull_ens_v1?.ef_1pct || "-"}</p>
                                        <p className="text-gray-700">AUPRC: {screening.grounded_win.h100_ehull_ens_v1?.auprc || "-"}</p>
                                        <p className="text-gray-700">Top 1% hit: {screening.grounded_win.h100_ehull_ens_v1?.top_1pct_hit_rate || "-"}</p>
                                    </div>
                                    <div className="border border-gray-200 rounded-lg p-4">
                                        <p className="font-medium text-gray-900 mb-2">OQMD Ensemble</p>
                                        <p className="text-gray-700">EF@1%: {screening.grounded_win.oqmd_ens_v1?.ef_1pct || "-"}</p>
                                        <p className="text-gray-700">AUPRC: {screening.grounded_win.oqmd_ens_v1?.auprc || "-"}</p>
                                        <p className="text-gray-700">Top 1% hit: {screening.grounded_win.oqmd_ens_v1?.top_1pct_hit_rate || "-"}</p>
                                    </div>
                                    <div className="border border-gray-200 rounded-lg p-4">
                                        <p className="font-medium text-gray-900 mb-2">JARVIS Ensemble</p>
                                        <p className="text-gray-700">EF@1%: {screening.grounded_win.jarvis_ens_v1?.ef_1pct || "-"}</p>
                                        <p className="text-gray-700">AUPRC: {screening.grounded_win.jarvis_ens_v1?.auprc || "-"}</p>
                                        <p className="text-gray-700">Top 1% hit: {screening.grounded_win.jarvis_ens_v1?.top_1pct_hit_rate || "-"}</p>
                                    </div>
                                </div>
                            </div>

                            <div className="border border-gray-200 rounded-lg overflow-hidden">
                                <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
                                    <h3 className="text-lg font-semibold text-gray-900">Proof Artifacts</h3>
                                </div>
                                <table className="w-full text-sm">
                                    <thead className="bg-white border-b border-gray-200">
                                        <tr>
                                            <th className="px-4 py-3 text-left font-medium text-gray-600">Artifact</th>
                                            <th className="px-4 py-3 text-left font-medium text-gray-600">Path</th>
                                            <th className="px-4 py-3 text-left font-medium text-gray-600">Size</th>
                                            <th className="px-4 py-3 text-left font-medium text-gray-600">Status</th>
                                            <th className="px-4 py-3 text-right font-medium text-gray-600">Action</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {screening.proofs.map((proof, idx) => (
                                            <tr key={proof.id} className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                                                <td className="px-4 py-3 text-gray-900">{PROOF_LABELS[proof.id] || proof.id}</td>
                                                <td className="px-4 py-3 font-mono text-xs text-gray-700">{proof.relative_path}</td>
                                                <td className="px-4 py-3 text-gray-700">{formatBytes(proof.size_bytes)}</td>
                                                <td className="px-4 py-3">
                                                    <span className={`text-xs font-medium ${proof.exists ? "text-green-700" : "text-red-700"}`}>
                                                        {proof.exists ? "present" : "missing"}
                                                    </span>
                                                </td>
                                                <td className="px-4 py-3 text-right">
                                                    <button
                                                        type="button"
                                                        onClick={() => downloadProof(proof)}
                                                        disabled={!proof.exists || !proof.download_url}
                                                        className="px-3 py-1 border border-gray-300 rounded text-xs hover:bg-gray-100 disabled:opacity-50"
                                                    >
                                                        Download
                                                    </button>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            {proofError ? <p className="text-sm text-red-600 mt-3">{proofError}</p> : null}
                        </>
                    ) : null}
                </section>
            </main>

            <footer className="border-t border-gray-200 mt-16">
                <div className="max-w-6xl mx-auto px-6 py-6">
                    <div className="flex justify-between text-sm text-gray-500 mb-3">
                        <p>CathodeScreen - Battery Discovery Platform</p>
                        <div className="flex gap-4">
                            <a href="https://github.com/ErenAri" target="_blank" className="hover:text-gray-900">GitHub</a>
                            <a href={`${API_BASE_URL}/docs`} target="_blank" className="hover:text-gray-900">API</a>
                            <Link href="/about" className="hover:text-gray-900">About</Link>
                        </div>
                    </div>
                    <p className="text-xs text-gray-400 text-center border-t border-gray-100 pt-3">
                        For screening purposes only. DFT validation recommended before synthesis. Model optimized for Li-oxide cathodes.
                    </p>
                </div>
            </footer>
        </div>
    );
}
