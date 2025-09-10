import { Link } from 'react-router-dom'

function Resources() {
  return (
    <div className="landing-root">
      <header className="landing-header glass fade-in-up">
        <div className="landing-header-left">
          <div className="logo-circle">💧</div>
          <span className="brand">Ground Water Companion</span>
        </div>
        <nav className="landing-nav">
          <Link to="/">Home</Link>
          <a href="#resources">Resources</a>
        </nav>
      </header>

      <section className="res-hero" id="resources">
        <div className="res-hero-inner">
          <h1>Resources</h1>
          <p>Find tools, ideas, and ready-to-go activities to bring groundwater concepts to life.</p>
        </div>
      </section>

      <main className="res-content">
        <section className="res-tiles">
          <article className="res-tile glass fade-in-up">
            <h3>At Home</h3>
            <p>Simple activities you can try at home to understand groundwater and conservation.</p>
            <button className="btn-primary" type="button">Explore</button>
          </article>
          <article className="res-tile glass fade-in-up" style={{ animationDelay: '120ms' }}>
            <h3>In Your Community</h3>
            <p>Classroom and community projects for group learning and field exploration.</p>
            <button className="btn-primary" type="button">Explore</button>
          </article>
        </section>
      </main>

      <footer className="landing-footer glass">
        <div className="footer-left">
          <div className="logo-circle small">💧</div>
          <span className="brand small">Ground Water Companion</span>
        </div>
        <div className="footer-right">
          <Link to="/">Home</Link>
          <a href="#resources">Resources</a>
          <a href="mailto:support@example.com">Support</a>
        </div>
      </footer>
    </div>
  )
}

export default Resources


