import { useEffect, useMemo, useState } from 'react';
import './App.css';

const CATEGORY_OPTIONS = [
  { value: 'all', label: 'All Categories' },
  { value: 'agentic', label: 'Agentic AI & MCP' },
  { value: 'rag', label: 'RAG & ML' },
  { value: 'trading', label: 'Trading & Analytics' },
  { value: 'research', label: 'Research & Innovation' },
];

const SORT_OPTIONS = [
  { value: 'wow', label: 'Wow Factor (High to Low)' },
  { value: 'recent', label: 'Most Recent' },
  { value: 'name', label: 'Name (A-Z)' },
];

const CATEGORY_BADGES = {
  agentic: 'bg-purple-600',
  rag: 'bg-green-600',
  trading: 'bg-amber-500',
  research: 'bg-blue-600',
  unknown: 'bg-slate-500',
};

const emptyStats = {
  total: 0,
  with_demo: 0,
  avg_wow_factor: 0,
  by_category: {
    agentic: 0,
    rag: 0,
    trading: 0,
    research: 0,
  },
};

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.statusText}`);
  }
  return response.json();
}

function App() {
  const [projects, setProjects] = useState([]);
  const [stats, setStats] = useState(emptyStats);
  const [filters, setFilters] = useState({
    category: 'all',
    sort: 'wow',
    search: '',
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      setLoading(true);
      setError('');

      const [projectsData, statsData] = await Promise.all([
        fetchJson('/api/projects'),
        fetchJson('/api/stats'),
      ]);

      setProjects(projectsData);
      setStats({
        ...emptyStats,
        ...statsData,
        by_category: {
          ...emptyStats.by_category,
          ...(statsData.by_category || {}),
        },
      });
      setLastUpdated(new Date());
    } catch (err) {
      console.error(err);
      setError(err.message || 'Unexpected error while loading projects');
    } finally {
      setLoading(false);
    }
  }

  const filteredProjects = useMemo(() => {
    const { category, sort, search } = filters;
    const term = search.trim().toLowerCase();

    const matches = projects.filter((project) => {
      const matchesCategory = category === 'all' || project.category === category;
      const matchesSearch =
        !term ||
        project.name.toLowerCase().includes(term) ||
        (project.description || '').toLowerCase().includes(term) ||
        (project.tech_stack || [])
          .join(' ')
          .toLowerCase()
          .includes(term);

      return matchesCategory && matchesSearch;
    });

    return matches.sort((a, b) => {
      if (sort === 'wow') {
        return (b.wow_factor || 0) - (a.wow_factor || 0);
      }
      if (sort === 'recent') {
        return new Date(b.created_at) - new Date(a.created_at);
      }
      if (sort === 'name') {
        return a.name.localeCompare(b.name);
      }
      return 0;
    });
  }, [projects, filters]);

  function handleFilterChange(field, value) {
    setFilters((prev) => ({
      ...prev,
      [field]: value,
    }));
  }

  return (
    <div className="min-h-screen">
      <header className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white py-10 px-4 shadow-lg">
        <div className="mx-auto flex max-w-7xl flex-col gap-3">
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-white/80">
            Anthropic Hackathon
          </p>
          <h1 className="text-4xl font-bold md:text-5xl">
            Meta-Builder Gallery
          </h1>
          <p className="text-lg text-white/90 md:w-3/4">
            Autonomous AI agents generating hackathon-grade prototypes with
            real demos. Explore the latest builds, filter by category, and dive
            into live projects.
          </p>
          {lastUpdated && (
            <p className="text-sm text-white/70">
              Last refreshed {lastUpdated.toLocaleString()}
            </p>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8">
        {error && (
          <div className="mb-6 rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
            <p className="font-semibold">Unable to load data</p>
            <p className="text-sm">{error}</p>
          </div>
        )}

        <StatsPanel stats={stats} loading={loading} />

        <FiltersBar
          filters={filters}
          onChange={handleFilterChange}
          onRefresh={loadData}
          loading={loading}
        />

        {loading ? (
          <div className="flex items-center justify-center py-24 text-gray-500">
            Fetching the latest builds...
          </div>
        ) : filteredProjects.length === 0 ? (
          <div className="rounded-xl bg-white p-12 text-center shadow">
            <div className="text-6xl text-indigo-200">⚙️</div>
            <h2 className="mt-4 text-2xl font-semibold text-gray-800">
              No projects match your filters
            </h2>
            <p className="mt-2 text-gray-500">
              Adjust the filters or create a new project with the orchestrator
              agent.
            </p>
          </div>
        ) : (
          <section className="grid gap-6 pb-12 md:grid-cols-2 lg:grid-cols-3">
            {filteredProjects.map((project) => (
              <ProjectCard key={project.id} project={project} />
            ))}
          </section>
        )}
      </main>
    </div>
  );
}

function StatsPanel({ stats, loading }) {
  const cards = [
    {
      label: 'Total Projects',
      value: stats.total,
      className:
        'bg-gradient-to-br from-indigo-500 to-purple-500 text-white shadow-lg',
    },
    {
      label: 'Agentic AI & MCP',
      value: stats.by_category.agentic || 0,
      className: 'border-2 border-purple-200 bg-purple-50 text-purple-700',
    },
    {
      label: 'RAG & ML',
      value: stats.by_category.rag || 0,
      className: 'border-2 border-green-200 bg-green-50 text-green-700',
    },
    {
      label: 'Trading & Analytics',
      value: stats.by_category.trading || 0,
      className: 'border-2 border-amber-200 bg-amber-50 text-amber-700',
    },
  ];

  return (
    <section className="grid gap-6 pb-6 md:grid-cols-4">
      {cards.map((card) => (
        <article
          key={card.label}
          className={`rounded-xl p-6 transition ${card.className}`}
        >
          <p className="text-sm font-semibold uppercase tracking-wide">
            {card.label}
          </p>
          <p className="mt-2 text-3xl font-bold">
            {loading ? '—' : card.value}
          </p>
        </article>
      ))}
    </section>
  );
}

function FiltersBar({ filters, onChange, onRefresh, loading }) {
  return (
    <section className="mb-8 rounded-xl bg-white p-6 shadow">
      <div className="flex flex-wrap gap-4">
        <label className="flex-1 min-w-[200px]">
          <p className="mb-1 text-sm font-medium text-gray-600">
            Filter by Category
          </p>
          <select
            className="w-full rounded-md border border-gray-300 px-3 py-2"
            value={filters.category}
            onChange={(event) => onChange('category', event.target.value)}
          >
            {CATEGORY_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <label className="flex-1 min-w-[200px]">
          <p className="mb-1 text-sm font-medium text-gray-600">Sort By</p>
          <select
            className="w-full rounded-md border border-gray-300 px-3 py-2"
            value={filters.sort}
            onChange={(event) => onChange('sort', event.target.value)}
          >
            {SORT_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <label className="flex-1 min-w-[200px]">
          <p className="mb-1 text-sm font-medium text-gray-600">Search</p>
          <input
            type="text"
            placeholder="Search projects..."
            className="w-full rounded-md border border-gray-300 px-3 py-2"
            value={filters.search}
            onChange={(event) => onChange('search', event.target.value)}
          />
        </label>

        <div className="flex-1 min-w-[150px]">
          <p className="mb-1 text-sm font-medium text-transparent">Refresh</p>
          <button
            className="w-full rounded-md bg-indigo-600 px-4 py-2 font-semibold text-white transition hover:bg-indigo-700 disabled:opacity-60"
            onClick={onRefresh}
            disabled={loading}
          >
            Refresh
          </button>
        </div>
      </div>
    </section>
  );
}

function ProjectCard({ project }) {
  const badgeClass = CATEGORY_BADGES[project.category] || CATEGORY_BADGES.unknown;
  const wowStars = '★'.repeat(Math.min(project.wow_factor || 0, 10));

  return (
    <article
      className="project-card flex flex-col rounded-2xl bg-white p-6 shadow transition hover:-translate-y-1 hover:shadow-xl"
      onClick={() => {
        window.location.href = `/project/${project.id}`;
      }}
    >
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-xl font-semibold text-gray-900">{project.name}</h3>
        <span
          className={`category-badge rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide text-white ${badgeClass}`}
        >
          {project.category || 'Unknown'}
        </span>
      </div>

      <p className="mt-3 text-sm text-gray-600">{project.description}</p>

      <div className="mt-4 flex flex-wrap gap-2">
        {(project.tech_stack || []).map((tech) => (
          <span
            key={tech}
            className="rounded-full bg-gray-100 px-3 py-1 text-xs font-medium text-gray-600"
          >
            {tech}
          </span>
        ))}
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-4 text-sm">
        <div>
          <dt className="text-gray-500">Wow Factor</dt>
          <dd className="wow-stars text-lg">{wowStars || '—'}</dd>
        </div>
        <div>
          <dt className="text-gray-500">Complexity</dt>
          <dd className="font-medium text-gray-800">
            {project.complexity || 'Unknown'}
          </dd>
        </div>
        <div>
          <dt className="text-gray-500">Created</dt>
          <dd className="font-medium text-gray-800">
            {project.created_at
              ? new Date(project.created_at).toLocaleDateString()
              : 'Unknown'}
          </dd>
        </div>
        <div>
          <dt className="text-gray-500">Demo</dt>
          <dd className="font-medium text-gray-800">
            {project.has_demo ? 'Available' : 'Not available'}
          </dd>
        </div>
      </dl>

      {project.has_readme && (
        <p className="mt-3 text-sm font-medium text-indigo-600">
          README available
        </p>
      )}
    </article>
  );
}

export default App;
