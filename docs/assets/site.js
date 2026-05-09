document.documentElement.classList.add("js");

const prefersReducedMotion = window.matchMedia(
  "(prefers-reduced-motion: reduce)"
).matches;

/* -------------------------------------------------------------------------
 * Mobile navigation
 * ------------------------------------------------------------------------- */
const navToggle = document.querySelector("[data-nav-toggle]");
const navLinksEl = document.querySelector("[data-nav-links]");

if (navToggle && navLinksEl) {
  const closeNav = () => {
    navLinksEl.classList.remove("is-open");
    navToggle.setAttribute("aria-expanded", "false");
    navToggle.setAttribute("aria-label", "Open navigation menu");
  };

  navToggle.addEventListener("click", () => {
    const isOpen = navLinksEl.classList.toggle("is-open");
    navToggle.setAttribute("aria-expanded", String(isOpen));
    navToggle.setAttribute(
      "aria-label",
      isOpen ? "Close navigation menu" : "Open navigation menu"
    );
  });

  navLinksEl.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", closeNav);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && navLinksEl.classList.contains("is-open")) {
      closeNav();
      navToggle.focus();
    }
  });
}

/* -------------------------------------------------------------------------
 * Active-link tracking
 * ------------------------------------------------------------------------- */
const header = document.querySelector("[data-site-header]");
const navAnchors = Array.from(document.querySelectorAll(".nav-links a"));

if (header && navAnchors.length > 0) {
  const sections = navAnchors
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (!visible) return;

      navAnchors.forEach((link) => {
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

/* -------------------------------------------------------------------------
 * BibTeX copy
 * ------------------------------------------------------------------------- */
const copyButton = document.querySelector("[data-copy-citation]");
const copyLabel = copyButton?.querySelector("[data-copy-label]");
const citation = document.querySelector("#bibtex");

if (copyButton && citation) {
  copyButton.addEventListener("click", async () => {
    const setLabel = (text) => {
      if (copyLabel) copyLabel.textContent = text;
      else copyButton.textContent = text;
    };

    try {
      await navigator.clipboard.writeText(citation.textContent.trim());
      setLabel("Copied");
      copyButton.classList.add("is-copied");
    } catch {
      setLabel("Select text");
    }

    window.setTimeout(() => {
      setLabel("Copy");
      copyButton.classList.remove("is-copied");
    }, 1800);
  });
}

/* -------------------------------------------------------------------------
 * Comparison sliders (drag + range fallback)
 * ------------------------------------------------------------------------- */
document.querySelectorAll("[data-comparison]").forEach((comparison) => {
  const range = comparison.querySelector(".comparison-range");
  const view = comparison.querySelector(".comparison-view");

  if (!range || !view) return;

  const setPos = (percent) => {
    const clamped = Math.min(100, Math.max(0, percent));
    view.style.setProperty("--pos", `${clamped}%`);
    range.value = String(clamped);
  };

  range.addEventListener("input", () => setPos(Number(range.value)));

  let dragging = false;

  const updateFromEvent = (event) => {
    const rect = view.getBoundingClientRect();
    const clientX =
      event.touches && event.touches.length ? event.touches[0].clientX : event.clientX;
    if (clientX === undefined) return;
    const percent = ((clientX - rect.left) / rect.width) * 100;
    setPos(percent);
  };

  view.addEventListener("pointerdown", (event) => {
    dragging = true;
    view.setPointerCapture?.(event.pointerId);
    updateFromEvent(event);
  });

  view.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    updateFromEvent(event);
  });

  const endDrag = (event) => {
    if (!dragging) return;
    dragging = false;
    if (event && event.pointerId !== undefined) {
      view.releasePointerCapture?.(event.pointerId);
    }
  };

  view.addEventListener("pointerup", endDrag);
  view.addEventListener("pointercancel", endDrag);
  view.addEventListener("pointerleave", endDrag);

  setPos(Number(range.value) || 50);
});

/* -------------------------------------------------------------------------
 * Toggle panels (more bald / more transfer)
 * ------------------------------------------------------------------------- */
document.querySelectorAll("[data-toggle-panel]").forEach((button) => {
  const panel = document.getElementById(button.dataset.togglePanel);
  if (!panel) return;

  const labelTarget = button.querySelector("[data-toggle-text]") || button;
  const closedLabel = button.dataset.closedLabel || labelTarget.textContent;
  const openLabel = button.dataset.openLabel || "Hide";

  button.addEventListener("click", () => {
    const willOpen = panel.hidden;
    panel.hidden = !willOpen;
    button.setAttribute("aria-expanded", String(willOpen));
    labelTarget.textContent = willOpen ? openLabel : closedLabel;
  });
});

/* -------------------------------------------------------------------------
 * Abstract expand/collapse
 * ------------------------------------------------------------------------- */
const abstractToggle = document.querySelector("[data-abstract-toggle]");
const abstractMore = document.querySelector("[data-abstract-more]");

if (abstractToggle && abstractMore) {
  const labelEl = abstractToggle.querySelector("[data-toggle-label]");
  const closedText = labelEl?.dataset.closedText || labelEl?.textContent || "Read full abstract";
  const openText = labelEl?.dataset.openText || "Show less";

  abstractToggle.addEventListener("click", () => {
    const willOpen = abstractMore.hidden;
    abstractMore.hidden = !willOpen;
    abstractToggle.setAttribute("aria-expanded", String(willOpen));
    if (labelEl) labelEl.textContent = willOpen ? openText : closedText;
  });
}

/* -------------------------------------------------------------------------
 * Carousel
 * ------------------------------------------------------------------------- */
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
      if (dotIndex === activeIndex) dot.setAttribute("aria-current", "true");
      else dot.removeAttribute("aria-current");
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
    frame = requestAnimationFrame(() => setActive(closestSlide()));
  });

  prev?.addEventListener("click", () => goTo(activeIndex - 1));
  next?.addEventListener("click", () => goTo(activeIndex + 1));

  dots.forEach((dot) => {
    dot.addEventListener("click", () => goTo(Number(dot.dataset.slide || 0)));
  });

  track.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      goTo(activeIndex - 1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      goTo(activeIndex + 1);
    }
  });

  setActive(0);
});

/* -------------------------------------------------------------------------
 * Reveal-on-scroll
 * ------------------------------------------------------------------------- */
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
