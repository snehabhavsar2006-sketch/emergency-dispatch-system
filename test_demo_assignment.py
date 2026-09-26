from environment import GridCity, Incident, Responder
from responder_manager import ResponderManager
from coordinator import Coordinator
from database import init_database, seed_demo_responders, get_responders


def test_demo_assignment_for_limited_pool():
    incident = Incident('INC-TEST', 'fire', (8, 8), 5)
    responders = [Responder('RESP-ONLY', 'police', (0, 0), 5, available=False)]
    coordinator = Coordinator(
        grid=GridCity(10, 10),
        responder_manager=ResponderManager(responders),
        hdc_top_n=3,
        seed=42,
    )
    trace = coordinator.dispatch_single(incident)
    assert trace.selected_responder is not None, 'Demo should still assign a responder'
    print('SELECTED', trace.selected_responder.responder_id)


def test_demo_responders_have_unique_addresses_and_demo_source():
    init_database()
    seed_demo_responders()
    rows = get_responders()
    assert len(rows) >= 5
    addresses = [row['address'] for row in rows if row.get('address')]
    assert len(addresses) == len(set(addresses))
    assert all(row.get('location_source', '').upper() == 'DEMO SIMULATION' for row in rows)


if __name__ == '__main__':
    test_demo_assignment_for_limited_pool()
    test_demo_responders_have_unique_addresses_and_demo_source()
