const header = document.querySelector("[data-site-header]");
const navLinks = Array.from(document.querySelectorAll(".nav-links a"));
const copyButton = document.querySelector("[data-copy-citation]");
const citation = document.querySelector("#bibtex");

if (header && navLinks.length > 0) {
  const sections = navLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (!visible) return;

      navLinks.forEach((link) => {
        link.classList.toggle(
          "is-active",
          link.getAttribute("href") === `#${visible.target.id}`
        );
      });
    },
    {
      rootMargin: "-22% 0px -58% 0px",
      threshold: [0.15, 0.3, 0.6],
    }
  );

  sections.forEach((section) => observer.observe(section));
}

if (copyButton && citation) {
  copyButton.addEventListener("click", async () => {
    const original = copyButton.textContent;

    try {
      await navigator.clipboard.writeText(citation.textContent.trim());
      copyButton.textContent = "Copied";
    } catch {
      copyButton.textContent = "Select text";
    }

    window.setTimeout(() => {
      copyButton.textContent = original;
    }, 1800);
  });
}
