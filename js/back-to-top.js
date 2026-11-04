/* Show a back-to-top button once the page has been scrolled. */
(function () {
  var btn = document.getElementById('back-to-top');
  if (!btn) return;

  function currentTop() {
    return Math.max(
      document.body.scrollTop || 0,
      document.documentElement.scrollTop || 0,
      window.pageYOffset || 0
    );
  }

  function update() {
    btn.classList.toggle('visible', currentTop() > 300);
  }

  document.addEventListener('scroll', update, { passive: true, capture: true });
  window.addEventListener('resize', update);

  btn.addEventListener('click', function () {
    if (document.body.scrollTop > 0) {
      document.body.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  });

  update();
})();
