document.documentElement.classList.add("js");

const prefersReducedMotion = window.matchMedia(
  "(prefers-reduced-motion: reduce)"
).matches;

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

document.querySelectorAll("[data-comparison]").forEach((comparison) => {
  const range = comparison.querySelector(".comparison-range");
  const view = comparison.querySelector(".comparison-view");

  if (!range || !view) return;

  const update = () => {
    view.style.setProperty("--pos", `${range.value}%`);
  };

  range.addEventListener("input", update);
  update();
});

document.querySelectorAll("[data-toggle-panel]").forEach((button) => {
  const panel = document.getElementById(button.dataset.togglePanel);

  if (!panel) return;

  const closedLabel = button.dataset.closedLabel || button.textContent;
  const openLabel = button.dataset.openLabel || "Hide";

  button.addEventListener("click", () => {
    const isOpen = !panel.hidden;

    panel.hidden = isOpen;
    button.setAttribute("aria-expanded", String(!isOpen));
    button.textContent = isOpen ? closedLabel : openLabel;
  });
});

document.querySelectorAll("[data-carousel]").forEach((carousel) => {
  const track = carousel.querySelector("[data-carousel-track]");
  const slides = Array.from(carousel.querySelectorAll(".transfer-slide"));
  const dots = Array.from(carousel.querySelectorAll("[data-carousel-dot]"));
  const prev = carousel.querySelector("[data-carousel-prev]");
  const next = carousel.querySelector("[data-carousel-next]");

  if (!track || slides.length === 0) return;

  let activeIndex = 0;
  let frame = null;

  const setActive = (index) => {
    activeIndex = Math.max(0, Math.min(index, slides.length - 1));
    dots.forEach((dot, dotIndex) => {
      if (dotIndex === activeIndex) {
        dot.setAttribute("aria-current", "true");
      } else {
        dot.removeAttribute("aria-current");
      }
    });
  };

  const closestSlide = () => {
    const trackRect = track.getBoundingClientRect();
    let closest = 0;
    let closestDistance = Number.POSITIVE_INFINITY;

    slides.forEach((slide, index) => {
      const slideRect = slide.getBoundingClientRect();
      const distance = Math.abs(slideRect.left - trackRect.left);

      if (distance < closestDistance) {
        closest = index;
        closestDistance = distance;
      }
    });

    return closest;
  };

  const goTo = (index) => {
    const targetIndex = Math.max(0, Math.min(index, slides.length - 1));

    slides[targetIndex].scrollIntoView({
      behavior: prefersReducedMotion ? "auto" : "smooth",
      block: "nearest",
      inline: "start",
    });
    setActive(targetIndex);
  };

  track.addEventListener("scroll", () => {
    if (frame) cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      setActive(closestSlide());
    });
  });

  prev?.addEventListener("click", () => goTo(activeIndex - 1));
  next?.addEventListener("click", () => goTo(activeIndex + 1));

  dots.forEach((dot) => {
    dot.addEventListener("click", () => {
      goTo(Number(dot.dataset.slide || 0));
    });
  });

  setActive(0);
});

const revealItems = Array.from(document.querySelectorAll(".reveal"));

if (prefersReducedMotion) {
  revealItems.forEach((item) => item.classList.add("is-visible"));
} else if (revealItems.length > 0) {
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;

        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      });
    },
    {
      rootMargin: "0px 0px -10% 0px",
      threshold: 0.12,
    }
  );

  revealItems.forEach((item) => revealObserver.observe(item));
}
