"""Rule-based personalised recommendations for BA Insight.
Each recommendation is triggered by a (satisfaction, aspect, sentiment)
combination. When a dissatisfied review mentions several aspects, the
aspect that passengers rate most negatively in the dataset as a whole
(mean VADER sentiment per aspect, measured on the 1,300 reviews) is
addressed first, so the recommendation matches the headline problem
in the review rather than a passing mention."""
RECOMMENDATIONS = {
    'Delays and punctuality': 'Proactively message affected passengers before they reach the airport, give a revised departure time and a clear compensation route; publish delay causes per route weekly.',
    'Baggage': 'Add live bag tracking in the BA app, staff a dedicated lost-bag desk at Heathrow, and auto-raise a compensation case when a bag is delayed over 12 hours.',
    'Seat comfort': 'Review seat pitch and cushioning on the worst-rated long-haul routes first; offer paid extra-legroom upgrades at check-in where seats score lowest.',
    'Food and drink': 'Refresh the economy menu on the routes with the lowest food scores, and give cabin crew a simple way to log stock-outs so catering can be fixed per flight.',
    'Staff service': 'Use the positive staff comments in crew briefings as examples of what works, and coach the specific behaviours named in negative reviews (tone at the gate, handling of complaints).',
    'Value for money': 'Make the fare difference explicit at booking (what is included vs low-cost rivals) and target upgrade offers at passengers who mention price.',
    'Booking and check-in': 'Fix the top online check-in failure modes, add a Heathrow queue-time display, and staff the bag-drop desks to match peak arrival patterns.',
    'In-flight entertainment': 'Prioritise screen repairs and content refreshes on long-haul routes, and offer a free Wi-Fi messaging tier where entertainment scores are lowest.',
    'Cleanliness': 'Add a turnaround cleanliness checklist signed off by the cabin manager, with extra attention to lavatories on long-haul flights.',
}
# Mean VADER sentiment per aspect across the 1,300 reviews (outputs/aspect_summary.csv).
ASPECT_PRIORITY = {
    'Delays and punctuality': -0.078, 'Baggage': -0.008, 'Value for money': 0.016,
    'Booking and check-in': 0.025, 'Seat comfort': 0.045, 'Cleanliness': 0.098,
    'Food and drink': 0.126, 'Staff service': 0.128, 'In-flight entertainment': 0.135,
}

def recommend(prediction):
    aspects = prediction.get('aspects', {})
    if prediction.get('predicted_satisfaction') != 'Dissatisfied':
        return 'No recovery action needed. Record what this passenger liked and use it as a positive example for the crew on this route.'
    candidates = [(a, s) for a, s in aspects.items() if s < 0.15 and a in RECOMMENDATIONS]
    if not candidates:
        return 'Apologise with a specific reference to the problem described, and offer a goodwill gesture tied to the journey affected.'
    candidates.sort(key=lambda kv: (ASPECT_PRIORITY.get(kv[0], 0.0), kv[1]))
    return RECOMMENDATIONS[candidates[0][0]]
