import { Link } from 'react-router-dom'
import AtHomeGW from './ProtectGW.jpg'

function AtHome() {
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
            <Link to="/resources" className="px-3 py-1.5 rounded-lg hover:bg-secondary-300/18 transition-colors">Resources</Link>
          </nav>
        </div>
      </header>

      <section className="bg-gradient-to-br from-secondary-300/85 to-secondary-400/75 py-12 px-4" id="at-home">
        <div className="w-full max-w-7xl mx-auto text-white">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold leading-tight mb-4">Way to conserve and protect groundwater</h1>
          <p className="text-lg sm:text-xl opacity-95">In this section, you'll find information on ways you can protect and conserve groundwater in and around your home.</p>
        </div>
      </section>

      <main className="flex-1 px-4 pb-6">
        <section className="w-full max-w-4xl mx-auto my-6">
          <article className="bg-gradient-card backdrop-blur-sm border border-dark-border rounded-2xl p-8 shadow-glass animate-float-up">
            <h3 className="text-2xl sm:text-3xl font-bold text-dark-text-primary mb-6">Top 10 Ways to Protect and Conserve</h3>
            <ul className="space-y-6 text-dark-text-primary">
              <li className="flex items-start gap-3">
                <span className="bg-primary-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 mt-1">1</span>
                <div>
                  <strong className="text-primary-300">Go Native</strong> – Use native plants in your landscape. They look great, and don't need much water or fertilizer. Also choose grass varieties for your lawn that are adapted for your region's climate, reducing the need for extensive watering or chemical applications.
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="bg-primary-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 mt-1">2</span>
                <div>
                  <strong className="text-primary-300">Reduce Chemical Use</strong> – Use fewer chemicals around your home and yard, and make sure to dispose of them properly – don't dump them on the ground!
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="bg-primary-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 mt-1">3</span>
                <div>
                  <strong className="text-primary-300">Manage Waste</strong> – Properly dispose of potentially toxic substances like unused chemicals, pharmaceuticals, paint, motor oil, and other substances. Many communities hold household hazardous waste collections or sites – contact your local health department to find one near you.
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="bg-primary-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 mt-1">4</span>
                <div>
                  <strong className="text-primary-300">Don't Let It Run</strong> – Shut off the water when you brush your teeth or shaving, and don't let it run while waiting for it to get cold. Keep a pitcher of cold water in the fridge instead.
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="bg-primary-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 mt-1">5</span>
                <div>
                  <strong className="text-primary-300">Fix the Drip</strong> – Check all the faucets, fixtures, toilets, and taps in your home for leaks and fix them right away, or install water conserving models.
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="bg-primary-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 mt-1">6</span>
                <div>
                  <strong className="text-primary-300">Wash Smarter</strong> – Limit yourself to just a five minute shower, and challenge your family members to do the same! Also, make sure to only run full loads in the dish and clothes washer.
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="bg-primary-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 mt-1">7</span>
                <div>
                  <strong className="text-primary-300">Water Wisely</strong> – Water the lawn and plants during the coolest parts of the day and only when they truly need it. Make sure you, your family, and your neighbors obey any watering restrictions during dry periods.
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="bg-primary-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 mt-1">8</span>
                <div>
                  <strong className="text-primary-300">Reduce, Reuse, and Recycle</strong> – Reduce the amount of "stuff" you use and reuse what you can. Recycle paper, plastic, cardboard, glass, aluminum and other materials.
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="bg-primary-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 mt-1">9</span>
                <div>
                  <strong className="text-primary-300">Natural Alternatives</strong> – Use all natural/nontoxic household cleaners whenever possible. Materials such as lemon juice, baking soda, and vinegar make great cleaning products, are inexpensive, and are environmentally friendly.
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="bg-primary-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 mt-1">10</span>
                <div>
                  <strong className="text-primary-300">Learn and Do More!</strong> – Get involved in water education! Learn more about groundwater and share your knowledge with others.
                </div>
              </li>
            </ul>
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
            <Link to="/resources" className="text-sm text-dark-text-primary hover:text-primary-300 transition-colors">Resources</Link>
            <a href="mailto:support@example.com" className="text-sm text-dark-text-primary hover:text-primary-300 transition-colors">Support</a>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default AtHome
