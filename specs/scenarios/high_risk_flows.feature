Feature: Clarify Gate bounded resolution

  Scenario: Vague-but-genuine input resolves within the round bound
    Given a user enters a vague goal like "I want to make apps"
    When the gate asks a narrowing question
    And the user answers with more specificity
    Then the gate either accepts a resolved role or asks one more narrowing question
    And the total narrowing rounds never exceed 2

  Scenario: User rejects the proposed interpretation twice
    Given the bounded rounds are exhausted
    And the system has proposed a best-guess role and the user rejected it
    And the system explained the role clearly and the user rejected it again
    When the grounding check runs on the user's own words
    Then if any market signal is found, the system proceeds at low confidence
    And if zero market signal is found, the system exits and builds no outline

Feature: Confidence Ladder enforcement

  Scenario: Both sources agree
    Given Himalayas and Tavily both return the same skill for a role
    When the Research Agent cross-validates the result
    Then the outline item is tagged confidence "high"

  Scenario: No source returns usable data
    Given Himalayas and Tavily both fail or return nothing for a role
    And roles_cache has no entry for the role
    When the Research Agent attempts to ground the outline
    Then no outline item is created
    And the system reports the general-knowledge-only floor explicitly to the user

Feature: Outline content is never removed

  Scenario: User sustains a "behind" pace
    Given a user's rolling-window pace is sustained below the drift threshold
    When the Coaching Agent triggers a pacing adjustment
    Then the outline's topic list is unchanged
    And only the day-by-day delivery schedule is extended

Feature: Verification retry cap is exactly 3 attempts

  Scenario: User fails a question twice, passes on the third attempt
    Given a user answers a verification question incorrectly on attempt 1
    And a fresh regenerated question incorrectly on attempt 2
    When the user answers a fresh regenerated question correctly on attempt 3
    Then the question is marked passed at full credit
    And exactly 3 question-generation calls have occurred for that question slot, not 4
    And the first attempt counted as attempt 1, not as a call made before the retry loop began

  Scenario: User fails all 3 attempts
    Given a user answers incorrectly on attempts 1, 2, and 3 for the same question slot
    When the retry cap is reached
    Then the system teaches the answer inline, citing the source material
    And the question is marked passed at half credit, not left unresolved
    And no fourth question-generation attempt occurs
