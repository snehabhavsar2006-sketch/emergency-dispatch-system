import time
from database import init_database, seed_demo_responders, create_incident, add_message, get_incidents, get_responders
from environment import Incident
from coordinator import Coordinator
from responder_manager import ResponderManager
from environment import GridCity

init_database()
seed_demo_responders()

inc_id = 'INC-VERIFY-001'
create_incident(
    inc_id,
    'accident',
    'Accident near NMIMS Shirpur Main Gate. Two people are injured.',
    'NMIMS Shirpur Main Gate',
    2,
    8,
    4,
    time.time(),
    'Emergency Message',
    'RECEIVED',
    None,
    None,
)
add_message('MSG-VERIFY-001', inc_id, 'Accident near NMIMS Shirpur Main Gate. Two people are injured.', time.time(), 'Emergency Message')

rows = get_incidents()
print('INC_COUNT', len(rows))
print('HAS_INC', any(r['incident_id'] == inc_id for r in rows))
print('RESP_COUNT', len(get_responders()))

responders = [
    __import__('environment').Responder(r['responder_id'], r['type'], (int(r['latitude']), int(r['longitude'])), int(r['capacity']), bool(r['availability']))
    for r in get_responders()
]
manager = ResponderManager(responders)
grid = GridCity(10, 10)
grid.set_terrain_cost(4, 5, 3.0)
grid.set_terrain_cost(5, 6, 2.5)
coordinator = Coordinator(grid=grid, responder_manager=manager, hdc_top_n=3, seed=42)
incident = Incident(inc_id, 'accident', (2, 8), 4, timestamp=time.time(), source='Emergency Message', message='Accident near NMIMS Shirpur Main Gate. Two people are injured.', location_name='NMIMS Shirpur Main Gate')
trace = coordinator.dispatch_single(incident)
print('SELECTED', trace.selected_responder.responder_id if trace.selected_responder else None)
print('HDC', len(trace.hdc_shortlist), 'CSP', len(trace.csp_valid), 'ROUTE', len(trace.astar_costs))
