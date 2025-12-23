def calc_probability(P_disease, P_positive_given_disease, P_positive_given_no_disease, P_no_disease):
    # Calculate the denominator (Total Probability of Positive Test)
    # P(Positive) = P(Positive|Disease) * P(Disease) + P(Positive|No Disease) * P(No Disease)
    denominator = (P_positive_given_disease * P_disease) + (P_positive_given_no_disease * P_no_disease)

    if denominator == 0:
        print("Invalid probability values.")
        return None

    # Calculate the numerator
    # P(Positive|Disease) * P(Disease)
    numerator = P_positive_given_disease * P_disease

    # Calculate Posterior Probability
    # P(Disease|Positive)
    probability = numerator / denominator

    # Display with 4 decimal precision
    print(f"Probability of having disease given a positive test: {probability:.4f}")

    return probability
