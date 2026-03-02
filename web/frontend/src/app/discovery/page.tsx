"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import API_BASE_URL, { API_KEY } from "@/config";
import { SocialLinks } from "@/components/SocialLinks";

// --- Types ---

interface CampaignSummary {
    campaign_name: string;
    current_cycle: number;
    current_stage: string;
    completed_cycles: number;
    total_dft_queries: number;
    total_stable_found: number;
    pool_remaining: number;
    pool_labeled: number;
}

interface CycleRecord {
    cycle: number;
    started_at: string;
    completed_at: string | null;
    stage: string;
    candidates_selected: string[];
    n_stable_found: number;
    metrics: Record<string, number>;
}

interface CampaignDetail extends CampaignSummary {
    created_at: string;
    cycles: CycleRecord[];
    stages: string[];
}

interface CandidateRow {
    material_id: string;
    formula: string;
    pred_ehull: number;
    p_stable: number;
    uncertainty: string;
    action: string;
    confidence_interval: [number, number];
    rank: number;
}

// --- Helpers ---

const STAGE_LABELS: Record<string, string> = {
    screen: "Screen",
    select: "Select",
    submit_dft: "Submit DFT",
    wait_dft: "Wait DFT",
    ingest: "Ingest",
    retrain: "Retrain",
    report: "Report",
};

function formatDate(iso: string): string {
    try {
        return new Date(iso).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            year: "numeric",
        });
    } catch {
        return iso;
    }
}

// --- Component ---

export default function DiscoveryPage() {
    // Campaign list
    const [campaigns, setCampaigns] = useState<CampaignSummary[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Selected campaign detail
    const [selected, setSelected] = useState<string | null>(null);
    const [detail, setDetail] = useState<CampaignDetail | null>(null);
    const [detailLoading, setDetailLoading] = useState(false);
    const [detailError, setDetailError] = useState<string | null>(null);

    // Candidates
    const [candidates, setCandidates] = useState<CandidateRow[]>([]);
    const [candidatesTotal, setCandidatesTotal] = useState(0);
    const [candidatesLoading, setCandidatesLoading] = useState(false);
    const [candidatesPage, setCandidatesPage] = useState(0);
    const [search, setSearch] = useState("");
    const perPage = 50;

    // Screen action
    const [screening, setScreening] = useState(false);

    // Create campaign
    const [showCreate, setShowCreate] = useState(false);
    const [newName, setNewName] = useState("");
    const [creating, setCreating] = useState(false);

    const headers = useMemo(() => {
        const h: HeadersInit = {};
        if (API_KEY) h["X-API-Key"] = API_KEY;
        return h;
    }, []);

    // --- Fetch campaigns ---
    const fetchCampaigns = () => {
        setLoading(true);
        fetch(`${API_BASE_URL}/discovery/campaigns`, { headers })
            .then((res) => res.json())
            .then((result) => {
                if (result.success) setCampaigns(result.campaigns);
                else setError(result.error || "Failed to load campaigns");
                setLoading(false);
            })
            .catch(() => {
                setError("Failed to connect to server");
                setLoading(false);
            });
    };

    useEffect(() => {
        fetchCampaigns();
    }, [headers]);

    // --- Fetch campaign detail ---
    const fetchDetail = (name: string) => {
        setDetailLoading(true);
        setDetailError(null);
        fetch(`${API_BASE_URL}/discovery/campaigns/${name}`, { headers })
            .then((res) => res.json())
            .then((result) => {
                if (result.success) setDetail(result.campaign);
                else setDetailError(result.error || "Failed to load campaign");
                setDetailLoading(false);
            })
            .catch(() => {
                setDetailError("Failed to connect to server");
                setDetailLoading(false);
            });
    };

    // --- Fetch candidates ---
    const fetchCandidates = (name: string, offset: number) => {
        setCandidatesLoading(true);
        fetch(
            `${API_BASE_URL}/discovery/campaigns/${name}/candidates?limit=${perPage}&offset=${offset}`,
            { headers }
        )
            .then((res) => res.json())
            .then((result) => {
                if (result.success) {
                    setCandidates(result.candidates);
                    setCandidatesTotal(result.total);
                }
                setCandidatesLoading(false);
            })
            .catch(() => setCandidatesLoading(false));
    };

    // When selecting a campaign
    const selectCampaign = (name: string) => {
        setSelected(name);
        setCandidatesPage(0);
        setSearch("");
        fetchDetail(name);
        fetchCandidates(name, 0);
    };

    // Paginate
    useEffect(() => {
        if (selected) {
            fetchCandidates(selected, candidatesPage * perPage);
        }
    }, [candidatesPage]);

    // --- Create campaign ---
    const handleCreate = () => {
        if (!newName.trim()) return;
        setCreating(true);
        fetch(`${API_BASE_URL}/discovery/campaigns`, {
            method: "POST",
            headers: { ...headers, "Content-Type": "application/json" },
            body: JSON.stringify({ name: newName.trim() }),
        })
            .then((res) => res.json())
            .then((result) => {
                if (result.success) {
                    setShowCreate(false);
                    setNewName("");
                    fetchCampaigns();
                    selectCampaign(result.campaign.campaign_name);
                } else {
                    alert(result.detail || "Failed to create campaign");
                }
                setCreating(false);
            })
            .catch(() => {
                alert("Failed to connect to server");
                setCreating(false);
            });
    };

    // --- Run screening ---
    const handleScreen = () => {
        if (!selected) return;
        setScreening(true);
        fetch(`${API_BASE_URL}/discovery/campaigns/${selected}/screen`, {
            method: "POST",
            headers,
        })
            .then((res) => res.json())
            .then((result) => {
                if (result.success) {
                    setCandidates(result.candidates.slice(0, perPage));
                    setCandidatesTotal(result.candidates_screened);
                    setCandidatesPage(0);
                    fetchDetail(selected);
                } else {
                    alert(result.detail || "Screening failed");
                }
                setScreening(false);
            })
            .catch(() => {
                alert("Failed to connect to server");
                setScreening(false);
            });
    };

    // --- Filter candidates ---
    const filtered = search
        ? candidates.filter(
              (c) =>
                  c.material_id.toLowerCase().includes(search.toLowerCase()) ||
                  c.formula.toLowerCase().includes(search.toLowerCase())
          )
        : candidates;

    return (
        <div className="min-h-screen bg-white font-sans text-gray-900">
            {/* Header */}
            <header className="fixed w-full top-4 z-50 flex justify-center pointer-events-none">
                <div className="bg-white/80 backdrop-blur-md border border-white/50 shadow-lg rounded-full px-8 py-3 flex items-center gap-12 pointer-events-auto">
                    <Link href="/" className="font-bold text-gray-900 text-lg tracking-tight hover:text-blue-600 hover:scale-125 transition-all duration-200">
                        CathodeScreen
                    </Link>
                    <nav className="flex items-center gap-8 text-sm font-medium text-gray-600">
                        <Link href="/predict" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">Predict</Link>
                        <Link href="/database" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">Database</Link>
                        <Link href="/database#screening-pack" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">Screening</Link>
                        <Link href="/discovery" className="text-blue-600 hover:scale-125 transition-all duration-200">Discovery</Link>
                        <Link href="/about" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">About</Link>
                    </nav>
                    <div className="pl-8 border-l border-gray-200">
                        <a href={`${API_BASE_URL}/docs`} target="_blank" className="text-xs font-bold text-blue-600 hover:text-blue-700 hover:scale-125 transition-all duration-200 uppercase tracking-wide flex items-center gap-1">
                            API <span className="text-[10px]">↗</span>
                        </a>
                    </div>
                </div>
            </header>

            <main className="pt-28 pb-16 px-6 max-w-6xl mx-auto">
                <h1 className="text-3xl font-bold text-gray-900 mb-2">Discovery Engine</h1>
                <p className="text-gray-600 mb-8">
                    Active learning campaign manager. Create campaigns, screen candidates with the MACE ensemble, and track discovery cycles.
                </p>

                {/* ======== CAMPAIGN DETAIL VIEW ======== */}
                {selected && detail ? (
                    <div>
                        {/* Back button */}
                        <button
                            onClick={() => { setSelected(null); setDetail(null); fetchCampaigns(); }}
                            className="text-sm text-blue-600 hover:text-blue-700 mb-6 flex items-center gap-1"
                        >
                            ← All Campaigns
                        </button>

                        <div className="flex items-center justify-between mb-6">
                            <div>
                                <h2 className="text-2xl font-bold text-gray-900">{detail.campaign_name}</h2>
                                <p className="text-sm text-gray-500">Created {formatDate(detail.created_at)} · Cycle {detail.current_cycle}</p>
                            </div>
                            <button
                                onClick={handleScreen}
                                disabled={screening || detail.current_stage !== "screen"}
                                className="px-5 py-2.5 bg-blue-600 text-white text-sm font-semibold rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                {screening ? "Screening..." : "Run ML Screening"}
                            </button>
                        </div>

                        {/* Stats cards */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                            <div className="border border-gray-200 rounded-lg p-4">
                                <p className="text-2xl font-semibold text-gray-900">{detail.pool_remaining.toLocaleString()}</p>
                                <p className="text-sm text-gray-500">Pool Remaining</p>
                            </div>
                            <div className="border border-blue-200 rounded-lg p-4">
                                <p className="text-2xl font-semibold text-blue-600">{detail.pool_labeled.toLocaleString()}</p>
                                <p className="text-sm text-gray-500">Labeled (DFT)</p>
                            </div>
                            <div className="border border-green-200 rounded-lg p-4">
                                <p className="text-2xl font-semibold text-green-600">{detail.total_stable_found}</p>
                                <p className="text-sm text-gray-500">Stable Found</p>
                            </div>
                            <div className="border border-indigo-200 rounded-lg p-4">
                                <p className="text-2xl font-semibold text-indigo-600">{detail.total_dft_queries}</p>
                                <p className="text-sm text-gray-500">DFT Queries</p>
                            </div>
                        </div>

                        {/* Stage pipeline */}
                        <div className="mb-8">
                            <h3 className="text-sm font-semibold text-gray-700 mb-3 uppercase tracking-wide">Pipeline Stage</h3>
                            <div className="flex items-center gap-1">
                                {detail.stages.map((stage, i) => {
                                    const stageIdx = detail.stages.indexOf(detail.current_stage);
                                    const isCompleted = i < stageIdx;
                                    const isCurrent = i === stageIdx;
                                    return (
                                        <div key={stage} className="flex-1 flex flex-col items-center gap-1">
                                            <div
                                                className={`w-full h-2 rounded-full ${
                                                    isCompleted
                                                        ? "bg-green-500"
                                                        : isCurrent
                                                        ? "bg-blue-500"
                                                        : "bg-gray-200"
                                                }`}
                                            />
                                            <span
                                                className={`text-[10px] ${
                                                    isCurrent
                                                        ? "text-blue-700 font-bold"
                                                        : isCompleted
                                                        ? "text-green-700"
                                                        : "text-gray-400"
                                                }`}
                                            >
                                                {STAGE_LABELS[stage] || stage}
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Candidates table */}
                        <div className="mb-8">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-lg font-semibold text-gray-900">
                                    Ranked Candidates
                                    <span className="text-sm font-normal text-gray-500 ml-2">
                                        ({candidatesTotal.toLocaleString()} total)
                                    </span>
                                </h3>
                                <input
                                    type="text"
                                    placeholder="Search by ID or formula..."
                                    value={search}
                                    onChange={(e) => setSearch(e.target.value)}
                                    className="px-3 py-2 border border-gray-300 rounded-lg text-sm w-64 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                />
                            </div>

                            {candidatesLoading ? (
                                <p className="text-gray-500 py-8 text-center">Loading candidates...</p>
                            ) : filtered.length === 0 ? (
                                <p className="text-gray-500 py-8 text-center">
                                    {candidatesTotal === 0 ? "Run ML Screening to score candidates." : "No matching candidates."}
                                </p>
                            ) : (
                                <>
                                    <div className="overflow-x-auto border border-gray-200 rounded-lg">
                                        <table className="w-full text-sm">
                                            <thead className="bg-gray-50 border-b border-gray-200">
                                                <tr>
                                                    <th className="px-4 py-3 text-left font-semibold text-gray-700">Rank</th>
                                                    <th className="px-4 py-3 text-left font-semibold text-gray-700">Material ID</th>
                                                    <th className="px-4 py-3 text-left font-semibold text-gray-700">Formula</th>
                                                    <th className="px-4 py-3 text-right font-semibold text-gray-700">E<sub>hull</sub> (eV)</th>
                                                    <th className="px-4 py-3 text-right font-semibold text-gray-700">P(Stable)</th>
                                                    <th className="px-4 py-3 text-center font-semibold text-gray-700">Uncertainty</th>
                                                    <th className="px-4 py-3 text-center font-semibold text-gray-700">Action</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-gray-100">
                                                {filtered.map((c) => (
                                                    <tr key={c.material_id} className="hover:bg-gray-50">
                                                        <td className="px-4 py-2.5 text-gray-500">{c.rank}</td>
                                                        <td className="px-4 py-2.5">
                                                            <a
                                                                href={`https://materialsproject.org/materials/${c.material_id}`}
                                                                target="_blank"
                                                                className="text-blue-600 hover:underline"
                                                            >
                                                                {c.material_id}
                                                            </a>
                                                        </td>
                                                        <td className="px-4 py-2.5 font-medium">{c.formula}</td>
                                                        <td className="px-4 py-2.5 text-right font-mono">{c.pred_ehull.toFixed(4)}</td>
                                                        <td className="px-4 py-2.5 text-right font-mono">{(c.p_stable * 100).toFixed(1)}%</td>
                                                        <td className="px-4 py-2.5 text-center">
                                                            <span
                                                                className={`px-2 py-1 rounded text-xs font-medium ${
                                                                    c.uncertainty === "Low"
                                                                        ? "bg-green-100 text-green-700"
                                                                        : c.uncertainty === "Medium"
                                                                        ? "bg-yellow-100 text-yellow-700"
                                                                        : "bg-red-100 text-red-700"
                                                                }`}
                                                            >
                                                                {c.uncertainty}
                                                            </span>
                                                        </td>
                                                        <td className="px-4 py-2.5 text-center">
                                                            <span
                                                                className={`px-2 py-1 rounded text-xs font-bold ${
                                                                    c.action === "DFT"
                                                                        ? "bg-green-100 text-green-700"
                                                                        : c.action === "HOLD"
                                                                        ? "bg-yellow-100 text-yellow-700"
                                                                        : "bg-red-100 text-red-600"
                                                                }`}
                                                            >
                                                                {c.action}
                                                            </span>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>

                                    {/* Pagination */}
                                    {candidatesTotal > perPage && (
                                        <div className="flex items-center justify-between mt-4">
                                            <p className="text-sm text-gray-500">
                                                Showing {candidatesPage * perPage + 1}–
                                                {Math.min((candidatesPage + 1) * perPage, candidatesTotal)} of{" "}
                                                {candidatesTotal.toLocaleString()}
                                            </p>
                                            <div className="flex gap-2">
                                                <button
                                                    onClick={() => setCandidatesPage((p) => Math.max(0, p - 1))}
                                                    disabled={candidatesPage === 0}
                                                    className="px-3 py-1.5 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
                                                >
                                                    Previous
                                                </button>
                                                <button
                                                    onClick={() => setCandidatesPage((p) => p + 1)}
                                                    disabled={(candidatesPage + 1) * perPage >= candidatesTotal}
                                                    className="px-3 py-1.5 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
                                                >
                                                    Next
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                </>
                            )}
                        </div>

                        {/* Cycle history */}
                        {detail.cycles.length > 0 && (
                            <div>
                                <h3 className="text-lg font-semibold text-gray-900 mb-4">Cycle History</h3>
                                <div className="overflow-x-auto border border-gray-200 rounded-lg">
                                    <table className="w-full text-sm">
                                        <thead className="bg-gray-50 border-b border-gray-200">
                                            <tr>
                                                <th className="px-4 py-3 text-left font-semibold text-gray-700">Cycle</th>
                                                <th className="px-4 py-3 text-left font-semibold text-gray-700">Stage</th>
                                                <th className="px-4 py-3 text-left font-semibold text-gray-700">Started</th>
                                                <th className="px-4 py-3 text-left font-semibold text-gray-700">Completed</th>
                                                <th className="px-4 py-3 text-right font-semibold text-gray-700">Selected</th>
                                                <th className="px-4 py-3 text-right font-semibold text-gray-700">Stable Found</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-gray-100">
                                            {detail.cycles.map((cycle) => (
                                                <tr key={cycle.cycle} className="hover:bg-gray-50">
                                                    <td className="px-4 py-2.5 font-medium">{cycle.cycle}</td>
                                                    <td className="px-4 py-2.5">
                                                        <span className="px-2 py-1 bg-blue-50 text-blue-700 rounded text-xs font-medium">
                                                            {STAGE_LABELS[cycle.stage] || cycle.stage}
                                                        </span>
                                                    </td>
                                                    <td className="px-4 py-2.5 text-gray-500">{formatDate(cycle.started_at)}</td>
                                                    <td className="px-4 py-2.5 text-gray-500">
                                                        {cycle.completed_at ? formatDate(cycle.completed_at) : "—"}
                                                    </td>
                                                    <td className="px-4 py-2.5 text-right">{cycle.candidates_selected.length}</td>
                                                    <td className="px-4 py-2.5 text-right">{cycle.n_stable_found}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}
                    </div>
                ) : (
                    /* ======== CAMPAIGN LIST VIEW ======== */
                    <div>
                        {loading ? (
                            <p className="text-gray-500 py-12 text-center">Loading campaigns...</p>
                        ) : error ? (
                            <p className="text-red-600 py-12 text-center">{error}</p>
                        ) : (
                            <>
                                {/* Aggregate stats */}
                                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8">
                                    <div className="border border-gray-200 rounded-lg p-4">
                                        <p className="text-2xl font-semibold text-gray-900">{campaigns.length}</p>
                                        <p className="text-sm text-gray-500">Total Campaigns</p>
                                    </div>
                                    <div className="border border-blue-200 rounded-lg p-4">
                                        <p className="text-2xl font-semibold text-blue-600">
                                            {campaigns.reduce((s, c) => s + c.pool_labeled, 0).toLocaleString()}
                                        </p>
                                        <p className="text-sm text-gray-500">Materials Labeled</p>
                                    </div>
                                    <div className="border border-green-200 rounded-lg p-4">
                                        <p className="text-2xl font-semibold text-green-600">
                                            {campaigns.reduce((s, c) => s + c.total_stable_found, 0)}
                                        </p>
                                        <p className="text-sm text-gray-500">Stable Found</p>
                                    </div>
                                </div>

                                {/* Action bar */}
                                <div className="flex items-center justify-between mb-4">
                                    <h2 className="text-lg font-semibold text-gray-900">Campaigns</h2>
                                    <button
                                        onClick={() => setShowCreate(!showCreate)}
                                        className="px-4 py-2 bg-blue-600 text-white text-sm font-semibold rounded-lg hover:bg-blue-700 transition-colors"
                                    >
                                        + New Campaign
                                    </button>
                                </div>

                                {/* Create form */}
                                {showCreate && (
                                    <div className="border border-blue-200 bg-blue-50 rounded-lg p-4 mb-6">
                                        <h3 className="text-sm font-semibold text-blue-900 mb-3">Create New Campaign</h3>
                                        <div className="flex gap-3">
                                            <input
                                                type="text"
                                                placeholder="Campaign name (e.g. lithium-oxide-v1)"
                                                value={newName}
                                                onChange={(e) => setNewName(e.target.value)}
                                                onKeyDown={(e) => e.key === "Enter" && handleCreate()}
                                                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                                            />
                                            <button
                                                onClick={handleCreate}
                                                disabled={creating || !newName.trim()}
                                                className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50"
                                            >
                                                {creating ? "Creating..." : "Create"}
                                            </button>
                                            <button
                                                onClick={() => { setShowCreate(false); setNewName(""); }}
                                                className="px-4 py-2 text-gray-600 text-sm border border-gray-300 rounded-lg hover:bg-gray-50"
                                            >
                                                Cancel
                                            </button>
                                        </div>
                                        <p className="text-xs text-gray-500 mt-2">
                                            Creates a campaign from the MACE ensemble predictions pool ({">"}1,000 TMO cathode materials).
                                        </p>
                                    </div>
                                )}

                                {/* Campaigns table */}
                                {campaigns.length === 0 ? (
                                    <div className="text-center py-16 border border-dashed border-gray-300 rounded-lg">
                                        <p className="text-gray-500 mb-2">No campaigns yet.</p>
                                        <p className="text-sm text-gray-400">Click &quot;+ New Campaign&quot; to start discovering stable cathode materials.</p>
                                    </div>
                                ) : (
                                    <div className="overflow-x-auto border border-gray-200 rounded-lg">
                                        <table className="w-full text-sm">
                                            <thead className="bg-gray-50 border-b border-gray-200">
                                                <tr>
                                                    <th className="px-4 py-3 text-left font-semibold text-gray-700">Campaign</th>
                                                    <th className="px-4 py-3 text-center font-semibold text-gray-700">Cycle</th>
                                                    <th className="px-4 py-3 text-center font-semibold text-gray-700">Stage</th>
                                                    <th className="px-4 py-3 text-right font-semibold text-gray-700">Pool</th>
                                                    <th className="px-4 py-3 text-right font-semibold text-gray-700">Labeled</th>
                                                    <th className="px-4 py-3 text-right font-semibold text-gray-700">Stable</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-gray-100">
                                                {campaigns.map((c) => (
                                                    <tr
                                                        key={c.campaign_name}
                                                        onClick={() => selectCampaign(c.campaign_name)}
                                                        className="hover:bg-blue-50 cursor-pointer"
                                                    >
                                                        <td className="px-4 py-3 font-medium text-blue-600">{c.campaign_name}</td>
                                                        <td className="px-4 py-3 text-center">{c.current_cycle}</td>
                                                        <td className="px-4 py-3 text-center">
                                                            <span className="px-2 py-1 bg-blue-50 text-blue-700 rounded text-xs font-medium">
                                                                {STAGE_LABELS[c.current_stage] || c.current_stage}
                                                            </span>
                                                        </td>
                                                        <td className="px-4 py-3 text-right">{c.pool_remaining.toLocaleString()}</td>
                                                        <td className="px-4 py-3 text-right">{c.pool_labeled}</td>
                                                        <td className="px-4 py-3 text-right">{c.total_stable_found}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </>
                        )}

                        {detailLoading && (
                            <p className="text-gray-500 py-8 text-center">Loading campaign...</p>
                        )}
                    </div>
                )}
            </main>

            {/* Footer */}
            <footer className="bg-gray-50 border-t border-gray-200 py-12">
                <div className="max-w-7xl mx-auto px-6 text-center">
                    <p className="text-gray-500 mb-4">CathodeScreen · Enterprise Platform · 2026</p>
                    <div className="flex justify-center mb-6">
                        <SocialLinks />
                    </div>
                    <div className="flex justify-center gap-6 text-sm text-gray-600 mb-4">
                        <Link href="/about" className="hover:text-blue-600">About</Link>
                        <Link href="/predict" className="hover:text-blue-600">Predict</Link>
                        <Link href="/database" className="hover:text-blue-600">Database</Link>
                        <Link href="/discovery" className="hover:text-blue-600">Discovery</Link>
                        <Link href="/database#screening-pack" className="hover:text-blue-600">Screening</Link>
                    </div>
                    <p className="text-xs text-gray-400 border-t border-gray-200 pt-4">
                        For screening purposes only. DFT validation recommended before synthesis. Model optimized for Li-oxide cathodes.
                    </p>
                </div>
            </footer>
        </div>
    );
}
