import { Link, useNavigate } from 'react-router-dom'

function Resources() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col">
      <header className="m-4 p-3 rounded-xl bg-gradient-card backdrop-blur-sm border border-dark-border shadow-glass animate-float-up">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-secondary-300/35 to-accent-300/35 flex items-center justify-center">
              💧
            </div>
            <span className="font-bold text-dark-text-primary">Ground Water Companion</span>
          </div>
          <nav className="flex gap-2">
            <Link to="/" className="px-3 py-1.5 rounded-lg hover:bg-secondary-300/18 transition-colors">Home</Link>
            <a href="#resources" className="px-3 py-1.5 rounded-lg hover:bg-secondary-300/18 transition-colors">Resources</a>
          </nav>
        </div>
      </header>

      <section className="bg-gradient-to-br from-secondary-300/85 to-secondary-400/75 py-12 px-4" id="resources">
        <div className="w-full max-w-7xl mx-auto text-white">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold leading-tight mb-4">Resources</h1>
          <p className="text-lg sm:text-xl opacity-95">Find tools, ideas, and ready-to-go activities to bring groundwater concepts to life.</p>
        </div>
      </section>

      <main className="flex-1 px-4 pb-6">
        <section className="w-full max-w-7xl mx-auto my-6 grid lg:grid-cols-2 gap-4">
          <article className="bg-gradient-card backdrop-blur-sm border border-dark-border rounded-2xl p-6 shadow-glass animate-float-up">
            <h3 className="text-2xl font-bold text-dark-text-primary mb-4">At Home</h3>
            <p className="text-dark-text-secondary mb-6 leading-relaxed">
              Simple activities you can try at home to understand groundwater and conservation.
            </p>
            <button 
              className="px-6 py-3 bg-gradient-primary text-white font-semibold rounded-lg hover:shadow-lg hover:-translate-y-0.5 transition-all duration-200" 
              type="button" 
              onClick={() => navigate('/at-home')}
            >
              Explore
            </button>
          </article>
          <article className="bg-gradient-card backdrop-blur-sm border border-dark-border rounded-2xl p-6 shadow-glass animate-float-up" style={{ animationDelay: '120ms' }}>
            <h3 className="text-2xl font-bold text-dark-text-primary mb-4">In Your Community</h3>
            <p className="text-dark-text-secondary mb-6 leading-relaxed">
              Classroom and community projects for group learning and field exploration.
            </p>
            <button 
              className="px-6 py-3 bg-gradient-primary text-white font-semibold rounded-lg hover:shadow-lg hover:-translate-y-0.5 transition-all duration-200" 
              type="button" 
              onClick={() => navigate('/in-your-community')}
            >
              Explore
            </button>
          </article>
        </section>
      </main>

      <footer className="m-4 p-3 rounded-xl bg-gradient-card backdrop-blur-sm border border-dark-border shadow-glass">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-secondary-300/35 to-accent-300/35 flex items-center justify-center">
              💧
            </div>
            <span className="font-semibold text-sm text-dark-text-primary">Ground Water Companion</span>
          </div>
          <div className="flex items-center gap-3">
            <Link to="/" className="text-sm text-dark-text-primary hover:text-primary-300 transition-colors">Home</Link>
            <a href="#resources" className="text-sm text-dark-text-primary hover:text-primary-300 transition-colors">Resources</a>
            <a href="mailto:support@example.com" className="text-sm text-dark-text-primary hover:text-primary-300 transition-colors">Support</a>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default Resources


