import { Link } from 'react-router-dom'

function Groundwater() {
  return (
    <div className="landing-root">
      <header className="landing-header glass fade-in-up">
        <div className="landing-header-left">
          <div className="logo-circle">💧</div>
          <span className="brand">Ground Water Companion</span>
        </div>
        <nav className="landing-nav">
          <Link to="/">Home</Link>
          <a href="#basics">Basics</a>
        </nav>
      </header>

      <section className="gw-hero" id="basics">
        <div className="gw-hero-inner">
          <h1>Groundwater Basics</h1>
        </div>
      </section>

      <main className="gw-content">
        <section className="gw-slab">
          <div className="gw-slab-inner">
            <p>Over 70% of the Earth’s surface is covered in water.</p>
            <p>But of that water, just 1% is readily available for human use, and of that 1%, 99% of it is stored beneath our feet as groundwater.</p>
            <p>We all rely on groundwater in some way, so it’s important that we understand this vital resource.</p>
          </div>
        </section>

        <section className="gw-split">
          <div className="gw-text">
            <h2>What Is Groundwater?</h2>
            <p><strong>Groundwater</strong> is the water found underground in the cracks and spaces in soil, sand and rock. It is stored in and moves slowly through geologic formations of soil, sand and rocks called aquifers.</p>
            <p>Groundwater is used for drinking water by millions of people, and is critical for irrigation. Managing groundwater sustainably helps ensure availability during dry periods.</p>
          </div>
          <div className="gw-media">
            <img className="gw-img-photo" alt="Lake and grassland" src="https://images.unsplash.com/photo-1473448912268-2022ce9509d8?q=80&w=1400&auto=format&fit=crop" />
          </div>
        </section>

        <section className="gw-split reverse">
          <div className="gw-media">
            <img className="gw-img-diagram" alt="Hydrologic cycle diagram" src="https://i.pinimg.com/736x/6b/6a/8b/6b6a8b4023ae7cc701eba4721db8dcbb.jpg" />
          </div>
          <div className="gw-text">
            <h2>The Hydrologic Cycle</h2>
            <p>Water is always on the move. Since the earth was formed, it has been endlessly circulating through the <strong>hydrologic cycle</strong>.</p>
            <p>Groundwater is an important part of this continuous cycle as water evaporates, forms clouds, returns to earth as precipitation, and recharges aquifers.</p>
            <Link to="/" className="btn-primary-link"><span className="btn-primary">Back to home</span></Link>
          </div>
        </section>
      </main>

      <footer className="landing-footer glass">
        <div className="footer-left">
          <div className="logo-circle small">💧</div>
          <span className="brand small">Ground Water Companion</span>
        </div>
        <div className="footer-right">
          <Link to="/">Home</Link>
          <a href="#basics">Basics</a>
          <a href="mailto:support@example.com">Support</a>
        </div>
      </footer>
    </div>
  )
}

export default Groundwater


