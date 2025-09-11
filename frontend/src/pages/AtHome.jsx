import { Link } from 'react-router-dom'
import AtHomeGW from './ProtectGW.jpg'

function AtHome() {
  return (
    <div className="landing-root">
      <header className="landing-header glass fade-in-up">
        <div className="landing-header-left">
          <div className="logo-circle">💧</div>
          <span className="brand">Ground Water Companion</span>
        </div>
        <nav className="landing-nav">
          <Link to="/">Home</Link>
          <Link to="/resources">Resources</Link>
        </nav>
      </header>

      <section className="res-hero" id="at-home">
        <div className="res-hero-inner">
          <h1>Way to conserve and protect groundwater</h1>
          <p>In this section, you’ll find information on ways you can protect and conserve groundwater in and around your home.</p>
        </div>
      </section>

      <main className="res-content at-home-layout">
        <section className="res-tiles res-tiles-at-home">
          <article className="res-tile glass fade-in-up">
            <h3>Top 10 Ways to Protect and Conserve</h3>
            <ul>
              <li><b>Go Native</b> – Use native plants in your landscape. They look great, and don’t need much water or fertilizer. Also choose grass varieties for your lawn that are adapted for your region’s climate, reducing the need for extensive watering or chemical applications.</li>
              <li><b>Reduce Chemical Use</b> – Use fewer chemicals around your home and yard, and make sure to dispose of them properly – don’t dump them on the ground!</li>
              <li><b>Manage Waste</b> – Properly dispose of potentially toxic substances like unused chemicals, pharmaceuticals, paint, motor oil, and other substances. Many communities hold household hazardous waste collections or sites – contact your local health department to find one near you.</li>
              <li><b>Don’t Let It Run</b> – Shut off the water when you brush your teeth or shaving, and don’t let it run while waiting for it to get cold. Keep a pitcher of cold water in the fridge instead.</li>
              <li><b>Fix the Drip</b> – Check all the faucets, fixtures, toilets, and taps in your home for leaks and fix them right away, or install water conserving models.</li>
              <li><b>Wash Smarter</b> – Limit yourself to just a five minute shower, and challenge your family members to do the same! Also, make sure to only run full loads in the dish and clothes washer.</li>
              <li><b>Water Wisely</b> – Water the lawn and plants during the coolest parts of the day and only when they truly need it. Make sure you, your family, and your neighbors obey any watering restrictions during dry periods.</li>
              <li><b>Reduce, Reuse, and Recycle</b> – Reduce the amount of “stuff” you use and reuse what you can. Recycle paper, plastic, cardboard, glass, aluminum and other materials.</li>
              <li><b>Natural Alternatives</b> – Use all natural/nontoxic household cleaners whenever possible. Materials such as lemon juice, baking soda, and vinegar make great cleaning products, are inexpensive, and are environmentally friendly.</li>
              <li><b>Learn and Do More!</b> – Get involved in water education! Learn more about groundwater and share your knowledge with others.</li>
            </ul>
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
          <Link to="/resources">Resources</Link>
          <a href="mailto:support@example.com">Support</a>
        </div>
      </footer>
    </div>
  )
}

export default AtHome
